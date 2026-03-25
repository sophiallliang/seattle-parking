"""
pages/overview.py
Tab 1 — high-level summary of Belltown parking patterns.

"""

import streamlit as st
import plotly.express as px
import pandas as pd

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def render(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    st.header("Belltown Parking Overview")
    st.caption(
        f"Training data: {len(train_df):,} records  |  "
        f"Test data: {len(test_df):,} records"
    )

    # ── Metric cards ─────────────────────────────────────────
    avg_occ     = train_df["occ_rate"].mean()
    peak_hour   = int(train_df.groupby("hour")["occ_rate"].mean().idxmax())
    busiest_dow = int(train_df.groupby("dow")["occ_rate"].mean().idxmax())

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg occupancy (train)", f"{avg_occ:.0%}")
    c2.metric("Peak hour", f"{peak_hour}:00 – {peak_hour + 1}:00")
    c3.metric("Busiest day", DAYS[busiest_dow])
    st.caption("Based on 2023 full-year training data (Jan–Dec 2023, Belltown only)")

    st.markdown("---")

    # ── Daily trend (test period) ────────────────────────────
    daily = (
        test_df.groupby("date")["occ_rate"]
        .mean()
        .reset_index()
    )
    daily.columns = ["date", "avg_occ_rate"]
    # Filter out anomalous dates (data collection failures)
    daily = daily[daily["avg_occ_rate"] >= 0.05]

    fig_trend = px.line(
        daily, x="date", y="avg_occ_rate",
        labels={"avg_occ_rate": "Avg occupancy rate", "date": "Date"},
        title="Daily average occupancy (test period)",
    )
    fig_trend.update_layout(yaxis_tickformat=".0%", height=300)
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── By day and by hour ───────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        by_dow = (
            train_df.groupby("dow")["occ_rate"]
            .mean()
            .reset_index()
        )
        by_dow["day"] = by_dow["dow"].map(lambda x: DAYS[x])
        fig_dow = px.bar(
            by_dow, x="day", y="occ_rate",
            title="Avg occupancy by day of week",
            labels={"occ_rate": "Occupancy rate", "day": ""},
        )
        fig_dow.update_layout(yaxis_tickformat=".0%", height=300)
        st.plotly_chart(fig_dow, use_container_width=True)
        st.caption("Note: Sunday excluded — Seattle paid parking is free on Sundays.")

    with col2:
        by_hour = (
            train_df.groupby("hour")["occ_rate"]
            .mean()
            .reset_index()
        )
        fig_hour = px.bar(
            by_hour, x="hour", y="occ_rate",
            title="Avg occupancy by hour",
            labels={"occ_rate": "Occupancy rate", "hour": "Hour"},
        )
        fig_hour.update_layout(yaxis_tickformat=".0%", height=300)
        st.plotly_chart(fig_hour, use_container_width=True)