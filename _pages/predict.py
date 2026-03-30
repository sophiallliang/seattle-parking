"""
pages/predict.py
Tab 2 — interactive occupancy prediction.

"""

import datetime
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data_loader import build_input_row
from model import predict, MODEL_NAMES, quality_label

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CAT_COLOR = {"Low": "green", "Medium": "orange", "High": "red"}

MONTH_OPTIONS = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr",  5:"May",  6:"Jun",
    7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
}

# Weather presets: (temp_c, precip_mm)
WEATHER_OPTIONS = {
    "☀️ Sunny":      (18.0, 0.0),
    "🌥 Cloudy":     (12.0, 0.0),
    "🌧 Light rain": (10.0, 2.0),
    "⛈ Heavy rain": (8.0,  10.0),
}


def render(regressors: dict, reg_results: dict,
           train_df: pd.DataFrame, holdout_results: dict = None) -> None:
    st.header("Predict occupancy")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Input conditions")

        # Model selector at the top
        sel_model = st.selectbox(
            "Model",
            MODEL_NAMES,
            index=MODEL_NAMES.index("Random Forest"),
            help="Random Forest is recommended — it achieves the highest R² on this dataset",
        )

        st.markdown("---")

        sel_day  = st.selectbox("Day of week", DAYS)
        sel_hour = st.slider("Hour of day", 8, 19, 14, format="%d:00")

        current_month = datetime.datetime.now().month
        sel_month = st.selectbox(
            "Month",
            options=list(MONTH_OPTIONS.keys()),
            format_func=lambda x: MONTH_OPTIONS[x],
            index=current_month - 1,
        )

        block_names = st.session_state.get("block_names", ["Unknown"])
        sel_block   = st.selectbox("Parking block", block_names)

        st.markdown("---")

        sel_weather          = st.selectbox("Weather conditions", list(WEATHER_OPTIONS.keys()))
        sel_temp, sel_precip = WEATHER_OPTIONS[sel_weather]

        sel_holiday = st.checkbox("Public holiday")

    # ── Lookup historical means ──────────────────────────────
    dow_val = DAYS.index(sel_day)
    enc     = st.session_state.get("block_encoder")
    try:
        block_id = int(enc.transform([sel_block])[0]) if enc else 0
    except Exception:
        block_id = 0

    bf_mean_tbl      = st.session_state.get("bf_mean",      pd.DataFrame())
    bf_hour_mean_tbl = st.session_state.get("bf_hour_mean", pd.DataFrame())
    bf_dow_mean_tbl  = st.session_state.get("bf_dow_mean",  pd.DataFrame())
    global_mean      = float(train_df["occ_rate"].mean())

    def _lookup(tbl: pd.DataFrame, keys: dict, col: str) -> float:
        if tbl.empty:
            return global_mean
        mask = pd.Series([True] * len(tbl), index=tbl.index)
        for k, v in keys.items():
            mask = mask & (tbl[k] == v)
        rows = tbl[mask]
        return float(rows[col].values[0]) if not rows.empty else global_mean

    bf_mean      = _lookup(bf_mean_tbl,      {"block": sel_block},                   "blockface_mean")
    bf_hour_mean = _lookup(bf_hour_mean_tbl, {"block": sel_block, "hour": sel_hour}, "blockface_hour_mean")
    bf_dow_mean  = _lookup(bf_dow_mean_tbl,  {"block": sel_block, "dow":  dow_val},  "blockface_dow_mean")

    input_row = build_input_row(
        hour=sel_hour, dow=dow_val, month=sel_month,
        block_id=block_id,
        blockface_mean=bf_mean,
        blockface_hour_mean=bf_hour_mean,
        blockface_dow_mean=bf_dow_mean,
        is_holiday=int(sel_holiday),
        temp_c=sel_temp,
        precip_mm=sel_precip,
    )

    result    = predict(regressors, sel_model, input_row)
    pred_rate = result["rate"]
    pred_cat  = result["category"]
    color     = CAT_COLOR[pred_cat]

    # Use internal R² if available, otherwise fall back to external
    if holdout_results and sel_model in holdout_results:
        r2       = holdout_results[sel_model]["R²"]
        r2_label = "Model R² (internal)"
    else:
        r2       = reg_results[sel_model]["R²"]
        r2_label = "Model R² (external)"

    qlabel, _ = quality_label(r2)

    # ── Output panel ─────────────────────────────────────────
    with col_out:
        st.subheader("Prediction result")

        m1, m2 = st.columns(2)
        m1.metric("Predicted occupancy", f"{pred_rate:.0%}")
        m2.metric(
            r2_label,
            f"{r2}",
            help=(
                f"Internal validation (20% holdout): R² = {r2}\n\n"
                "Random Forest is the best model based on internal validation. "
                "The drop in external R² is expected due to the 3-year gap between training and test data."
            )
        )

        st.markdown(
            f"**Category:** :{color}[**{pred_cat} occupancy**]  "
            f"(confidence: {result['confidence']})"
        )

        # Capacity: use mode per block (most common = normal capacity)
        if "block" in train_df.columns and sel_block in train_df["block"].values:
            mode_vals = train_df[train_df["block"] == sel_block]["capacity"].mode()
            capacity  = int(mode_vals.iloc[0]) if not mode_vals.empty else int(train_df["capacity"].mode().iloc[0])
        else:
            capacity = int(train_df["capacity"].mode().iloc[0])

        available = max(0, round(capacity * (1 - pred_rate)))
        st.info(
            f"**Estimated available spaces:** {available} / {capacity}  "
            f"— based on historical avg capacity for this block"
        )

        if sel_holiday:
            st.warning("Public holiday — patterns may differ from normal days")
        if sel_precip > 0.5:
            st.info(f"{sel_weather} conditions typically increase occupancy")

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pred_rate * 100),
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0,  40], "color": "#d4edda"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70,100], "color": "#f8d7da"},
                ],
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption(
            f"Model: {sel_model}  |  "
            f"MAE: {reg_results[sel_model]['MAE']}  |  "
            f"Historical avg for this block: {bf_mean:.0%}"
        )