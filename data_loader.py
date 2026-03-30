"""
data_loader.py
Handles data loading, cleaning, hourly aggregation, and feature engineering.

"""

import pandas as pd
import numpy as np
import streamlit as st
import requests
import holidays
from sklearn.preprocessing import LabelEncoder

# ── Constants ────────────────────────────────────────────────
FEATURES = [
    "hour", "dow", "month",
    "is_weekend", "is_lunch", "is_evening", "is_fri_pm",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "block_id",
    "blockface_mean", "blockface_hour_mean", "blockface_dow_mean",
    "is_holiday", "is_holiday_eve",
    "temp_c", "precip_mm", "is_rainy",
]

OCC_BINS   = [-0.01, 0.4, 0.7, 1.01]
OCC_LABELS = ["Low", "Medium", "High"]

COL_MAP = {
    "OccupancyDateTime":        "datetime",
    "PaidOccupancy":            "occupied",
    "ParkingSpaceCount":        "capacity",
    "PaidParkingRate":          "rate",
    "BlockfaceName":            "block",
    "SideOfStreet":             "side",
    "ParkingTimeLimitCategory": "time_limit",
    "PaidParkingArea":          "area",
    "PaidParkingSubArea":       "subarea",
    "ParkingCategory":          "parking_cat",
    "Location":                 "location",
}

SAMPLE_PER_MONTH = 50_000   # increased from 30k for better coverage
SEATTLE_LAT      = 47.6062
SEATTLE_LON      = -122.3321


# ── Weather fetcher ──────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={SEATTLE_LAT}&longitude={SEATTLE_LON}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,precipitation"
        "&timezone=America%2FLos_Angeles"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()["hourly"]
        weather_df = pd.DataFrame({
            "datetime":  pd.to_datetime(data["time"]),
            "temp_c":    data["temperature_2m"],
            "precip_mm": data["precipitation"],
        })
        weather_df["date"] = weather_df["datetime"].dt.date
        weather_df["hour"] = weather_df["datetime"].dt.hour
        return weather_df[["date", "hour", "temp_c", "precip_mm"]]
    except Exception as e:
        st.warning(f"Weather data unavailable ({e}). Using default values.")
        return pd.DataFrame(columns=["date", "hour", "temp_c", "precip_mm"])


# ── Holiday builder ──────────────────────────────────────────
def _build_holiday_sets(years: list) -> tuple:
    us_holidays       = holidays.US(state="WA", years=years)
    holiday_dates     = set(us_holidays.keys())
    holiday_eve_dates = {
        (pd.Timestamp(d) - pd.Timedelta(days=1)).date()
        for d in holiday_dates
    }
    return holiday_dates, holiday_eve_dates


# ── Hourly aggregator ────────────────────────────────────────
def _aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate minute-level records to hourly level per blockface.
    occ_rate = mean(PaidOccupancy / ParkingSpaceCount) over the hour.
    """
    df = df[(df["occupied"] >= 0) & (df["capacity"] > 0)].copy()
    df["occ_rate"] = (df["occupied"] / df["capacity"]).clip(0, 1)
    df["hour"]     = df["datetime"].dt.hour
    df["dow"]      = df["datetime"].dt.dayofweek
    df["month"]    = df["datetime"].dt.month
    df["date"]     = df["datetime"].dt.date

    agg = df.groupby(["block", "date", "hour"]).agg(
        occ_rate=("occ_rate", "mean"),
        capacity=("capacity", "first"),
        dow     =("dow",      "first"),
        month   =("month",    "first"),
    ).reset_index()

    return agg


# ── Historical mean lookup builder ──────────────────────────
def _build_mean_lookup(df_full: pd.DataFrame) -> dict:
    """
    Compute historical mean features from the FULL aggregated dataset
    (before sampling) so the means are stable and representative.
    Returns a dict of lookup DataFrames for merging later.
    """
    # Overall mean per blockface
    bf_mean = (
        df_full.groupby("block")["occ_rate"]
        .mean()
        .reset_index()
        .rename(columns={"occ_rate": "blockface_mean"})
    )

    # Mean per blockface × hour
    bf_hour_mean = (
        df_full.groupby(["block", "hour"])["occ_rate"]
        .mean()
        .reset_index()
        .rename(columns={"occ_rate": "blockface_hour_mean"})
    )

    # Mean per blockface × dow
    bf_dow_mean = (
        df_full.groupby(["block", "dow"])["occ_rate"]
        .mean()
        .reset_index()
        .rename(columns={"occ_rate": "blockface_dow_mean"})
    )

    # Also store for prediction-time lookup
    st.session_state["bf_mean"]      = bf_mean
    st.session_state["bf_hour_mean"] = bf_hour_mean
    st.session_state["bf_dow_mean"]  = bf_dow_mean

    return {
        "bf_mean":      bf_mean,
        "bf_hour_mean": bf_hour_mean,
        "bf_dow_mean":  bf_dow_mean,
    }


# ── Stratified sampler ───────────────────────────────────────
def _stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Stratified sampling by (dow × time_slot).
    Preserves natural time distribution proportionally.
    """
    if len(df) <= n:
        return df

    df = df.copy()
    df["_time_slot"] = pd.cut(
        df["hour"],
        bins=[-1, 7, 11, 16, 20, 23],
        labels=["Early Morning", "Morning Peak", "Midday", "Evening Peak", "Night"],
    )
    df["_stratum"] = df["dow"].astype(str) + "_" + df["_time_slot"].astype(str)

    strata_counts = df["_stratum"].value_counts()
    total         = len(df)
    sampled_parts = []

    for stratum, count in strata_counts.items():
        stratum_df   = df[df["_stratum"] == stratum]
        n_from_group = max(1, round(n * count / total))
        n_from_group = min(n_from_group, len(stratum_df))
        sampled_parts.append(
            stratum_df.sample(n=n_from_group, random_state=42)
        )

    result = pd.concat(sampled_parts).drop(columns=["_time_slot", "_stratum"])
    if len(result) > n:
        result = result.sample(n=n, random_state=42)
    return result.reset_index(drop=True)


# ── Main loader ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Full pipeline:
      1. Chunked CSV read
      2. Hourly aggregation per blockface
      3. Build historical mean lookup from FULL data  ← key fix
      4. Stratified sample per month
      5. Merge historical means into sampled data
      6. Feature engineering + holiday + weather
    """

    # ── Step 1: Chunked read + bucket by month ───────────────
    month_frames: dict = {}

    with st.spinner("Reading CSV in chunks…"):
        for chunk in pd.read_csv(filepath, low_memory=False, chunksize=500_000):
            chunk = chunk.rename(
                columns={k: v for k, v in COL_MAP.items() if k in chunk.columns}
            )
            chunk["datetime"] = pd.to_datetime(chunk["datetime"], errors="coerce")
            chunk = chunk.dropna(subset=["datetime"])
            chunk["_month"] = chunk["datetime"].dt.month
            for m in chunk["_month"].unique():
                if m not in month_frames:
                    month_frames[m] = []
                month_frames[m].append(chunk[chunk["_month"] == m])

    # ── Step 2: Hourly aggregation ───────────────────────────
    with st.spinner("Aggregating to hourly level…"):
        agg_months = []
        for m, frames in month_frames.items():
            if not frames:
                continue
            month_raw = pd.concat(frames, ignore_index=True)
            month_agg = _aggregate_hourly(month_raw)
            agg_months.append(month_agg)
        df_full = pd.concat(agg_months, ignore_index=True)

    # Business hours filter on full data
    df_full = df_full[df_full["hour"].between(8, 19)].copy()


    # ── Step 3: Remove anomalous dates ──────────────────────
    # Dates where daily avg occ_rate < 0.01 are data collection failures
    with st.spinner("Removing anomalous dates…"):
        daily_avg   = df_full.groupby("date")["occ_rate"].mean()
        valid_dates = daily_avg[daily_avg >= 0.01].index
        df_full     = df_full[df_full["date"].isin(valid_dates)].copy()

    # ── Step 4: Build mean lookup from FULL aggregated data ──
    with st.spinner("Computing historical mean features…"):
        lookup = _build_mean_lookup(df_full)

    # ── Step 5: Stratified sample per month ─────────────────
    with st.spinner("Stratified sampling by month…"):
        sampled = []
        for m, grp in df_full.groupby("month"):
            sampled.append(_stratified_sample(grp, SAMPLE_PER_MONTH))
        df = pd.concat(sampled, ignore_index=True)

    # ── Step 5: Merge historical means into sampled data ────
    df = df.merge(lookup["bf_mean"],      on="block",           how="left")
    df = df.merge(lookup["bf_hour_mean"], on=["block", "hour"], how="left")
    df = df.merge(lookup["bf_dow_mean"],  on=["block", "dow"],  how="left")

    # Fill any missing with global mean
    global_mean = df_full["occ_rate"].mean()
    df["blockface_mean"]      = df["blockface_mean"].fillna(global_mean)
    df["blockface_hour_mean"] = df["blockface_hour_mean"].fillna(global_mean)
    df["blockface_dow_mean"]  = df["blockface_dow_mean"].fillna(global_mean)

    # ── Step 6: Time features ────────────────────────────────
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_lunch"]   = df["hour"].between(11, 13).astype(int)
    df["is_evening"] = df["hour"].between(17, 19).astype(int)
    df["is_fri_pm"]  = ((df["dow"] == 4) & df["hour"].between(14, 18)).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["dow"]  / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dow"]  / 7)

    # ── Step 7: Block encoding ───────────────────────────────
    le = LabelEncoder()
    df["block_id"] = le.fit_transform(df["block"].astype(str))
    st.session_state["block_encoder"] = le
    st.session_state["block_names"]   = list(le.classes_)

    # ── Step 8: Holiday flags ────────────────────────────────
    with st.spinner("Adding holiday flags…"):
        years = list(pd.to_datetime(df["date"].astype(str)).dt.year.unique())
        holiday_dates, holiday_eve_dates = _build_holiday_sets(years)
        date_ts = pd.to_datetime(df["date"].astype(str)).dt.date
        df["is_holiday"]     = date_ts.isin(holiday_dates).astype(int)
        df["is_holiday_eve"] = date_ts.isin(holiday_eve_dates).astype(int)

    # ── Step 9: Weather ──────────────────────────────────────
    with st.spinner("Fetching Seattle weather…"):
        start = str(df["date"].min())
        end   = str(df["date"].max())
        weather_df = _fetch_weather(start, end)

    if not weather_df.empty:
        df["date_key"]     = pd.to_datetime(df["date"].astype(str)).dt.date
        weather_df["date_key"] = weather_df["date"]
        df = df.merge(
            weather_df[["date_key", "hour", "temp_c", "precip_mm"]],
            on=["date_key", "hour"], how="left"
        )
        df["temp_c"]    = df["temp_c"].fillna(df["temp_c"].median())
        df["precip_mm"] = df["precip_mm"].fillna(0)
        df = df.drop(columns=["date_key"])
    else:
        df["temp_c"]    = 11.0
        df["precip_mm"] = 0.0

    df["is_rainy"] = (df["precip_mm"] > 0.5).astype(int)

    # ── Step 10: Occupancy category ──────────────────────────
    df["occ_cat"]  = pd.cut(df["occ_rate"], bins=OCC_BINS, labels=OCC_LABELS)
    df["day_type"] = df["is_weekend"].map({0: "Weekday", 1: "Weekend"})

    return df.dropna(subset=["occ_rate", "occ_cat"]).reset_index(drop=True)


# ── Prediction input builder ─────────────────────────────────
def build_input_row(hour: int, dow: int, month: int, block_id: int,
                    blockface_mean: float, blockface_hour_mean: float,
                    blockface_dow_mean: float,
                    is_holiday: int = 0, temp_c: float = 11.0,
                    precip_mm: float = 0.0) -> pd.DataFrame:
    """
    Build a single-row DataFrame for prediction.
    Historical means come from the full-data lookup stored in session_state.
    """
    row = {
        "hour":                hour,
        "dow":                 dow,
        "month":               month,
        "is_weekend":          int(dow >= 5),
        "is_lunch":            int(11 <= hour <= 13),
        "is_evening":          int(17 <= hour <= 19),
        "is_fri_pm":           int(dow == 4 and 14 <= hour <= 18),
        "hour_sin":            np.sin(2 * np.pi * hour / 24),
        "hour_cos":            np.cos(2 * np.pi * hour / 24),
        "dow_sin":             np.sin(2 * np.pi * dow  / 7),
        "dow_cos":             np.cos(2 * np.pi * dow  / 7),
        "block_id":            block_id,
        "blockface_mean":      blockface_mean,
        "blockface_hour_mean": blockface_hour_mean,
        "blockface_dow_mean":  blockface_dow_mean,
        "is_holiday":          is_holiday,
        "is_holiday_eve":      0,
        "temp_c":              temp_c,
        "precip_mm":           precip_mm,
        "is_rainy":            int(precip_mm > 0.5),
    }
    return pd.DataFrame([row])[FEATURES]