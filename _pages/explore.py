"""
pages/explore.py
Tab 3 — EDA visualizations and data insights.

"""

import streamlit as st
import plotly.express as px
import pandas as pd

DAYS_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def render(train_df: pd.DataFrame) -> None:
    st.header("Explore patterns")

    # ── Heatmap: hour × day of week ──────────────────────────
    pivot = (
        train_df.groupby(["dow", "hour"])["occ_rate"]
        .mean()
        .unstack()
    )
    pivot.index = [DAYS_SHORT[i] for i in pivot.index]

    fig_hm = px.imshow(
        pivot,
        labels={"x": "Hour", "y": "Day", "color": "Occupancy rate"},
        color_continuous_scale="YlOrRd",
        title="Occupancy heatmap — hour × day of week",
        aspect="auto",
        zmin=0, zmax=1,
    )
    fig_hm.update_layout(height=320)
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption("Note: Sunday is not shown — Seattle paid parking is free on Sundays and no occupancy data is collected.")

    # ── Distribution + Weekday vs Weekend ───────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig_dist = px.histogram(
            train_df, x="occ_rate", nbins=40, color="occ_cat",
            category_orders={"occ_cat": ["Low", "Medium", "High"]},
            color_discrete_map={
                "Low":    "#2d8a4e",
                "Medium": "#e6a817",
                "High":   "#c0392b",
            },
            title="Occupancy rate distribution",
            labels={"occ_rate": "Occupancy rate", "occ_cat": "Category"},
        )
        fig_dist.update_layout(height=320)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        by_type = (
            train_df.groupby(["hour", "day_type"])["occ_rate"]
            .mean()
            .reset_index()
        )
        fig_ww = px.line(
            by_type, x="hour", y="occ_rate", color="day_type",
            title="Weekday vs weekend — hourly profile",
            labels={
                "occ_rate": "Occupancy rate",
                "hour": "Hour",
                "day_type": "",
            },
        )
        fig_ww.update_layout(yaxis_tickformat=".0%", height=320)
        st.plotly_chart(fig_ww, use_container_width=True)

    # ── Top blocks ───────────────────────────────────────────
    if "block" in train_df.columns:
        st.subheader("Busiest parking blocks")
        top_blocks = (
            train_df.groupby("block")["occ_rate"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_blocks.columns = ["Block", "Avg occupancy rate"]
        top_blocks["Avg occupancy rate"] = top_blocks["Avg occupancy rate"].map(
            lambda x: f"{x:.1%}"
        )
        st.dataframe(top_blocks, use_container_width=True, hide_index=True)

    # ── Auto-generated insights ──────────────────────────────
    st.subheader("Key insights")

    high_df       = train_df[train_df["occ_cat"] == "High"]
    lunch_pct     = len(train_df[(train_df["is_lunch"] == 1) & (train_df["occ_cat"] == "High")]) \
                    / max(len(high_df), 1) * 100
    fri_pm_avg    = train_df[train_df["is_fri_pm"]  == 1]["occ_rate"].mean()
    late_avg      = train_df[train_df["hour"]       >= 20]["occ_rate"].mean()
    weekend_avg   = train_df[train_df["is_weekend"] == 1]["occ_rate"].mean()
    weekday_avg   = train_df[train_df["is_weekend"] == 0]["occ_rate"].mean()

    ic1, ic2, ic3 = st.columns(3)
    ic1.info(
        f"**Lunch peak** (11–13h) accounts for "
        f"**{lunch_pct:.0f}%** of high-occupancy records."
    )
    ic2.warning(
        f"**Friday afternoon** avg occupancy: **{fri_pm_avg:.0%}**"
    )
    ic3.success(
        f"**After 8pm** avg drops to **{late_avg:.0%}**"
    )

    st.caption(
        f"Weekday avg: {weekday_avg:.0%}  |  "
        f"Weekend avg: {weekend_avg:.0%}  |  "
        f"Overall avg: {train_df['occ_rate'].mean():.0%}"
    )