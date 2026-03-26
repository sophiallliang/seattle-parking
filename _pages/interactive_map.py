"""
pages/interactive_map.py
Advanced interactive map for global batch prediction, animation, pricing, and routing.
"""

import datetime
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import pandas as pd

from data_loader import build_input_row, FEATURES
from model import predict, MODEL_NAMES, quality_label

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CAT_COLOR = {"Low": "green", "Medium": "orange", "High": "red"}
MONTH_OPTIONS = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr",  5:"May",  6:"Jun",
    7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
}
WEATHER_OPTIONS = {
    "☀️ Sunny":      (18.0, 0.0),
    "🌥 Cloudy":     (12.0, 0.0),
    "🌧 Light rain": (10.0, 2.0),
    "⛈ Heavy rain": (8.0,  10.0),
}
LANDMARKS = {
    "None": None,
    "Space Needle": (47.6205, -122.3493),
    "Pike Place Market": (47.6097, -122.3422), 
    "Olympic Sculpture Park": (47.6166, -122.3553), 
    "Amazon Spheres": (47.6158, -122.3396)
}

def render(regressors: dict, clf, reg_results: dict, train_df: pd.DataFrame) -> None:
    st.header("Interactive Map")

    if "bf_loc" not in st.session_state or st.session_state["bf_loc"].empty:
        st.warning("Location data not available.")
        return
    if "bf_mean" not in st.session_state or st.session_state["bf_mean"].empty:
        st.warning("Historical mean data not available.")
        return

    # ── 1. Advanced Settings Panel ───────────────────────────────
    with st.expander("Advanced Map Settings (Time, Filter, Routing)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            sel_model = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index("Random Forest") if "Random Forest" in MODEL_NAMES else 0, key="map_model_global")
            sel_day  = st.selectbox("Day of week", DAYS, key="map_day_global")
            sel_landmark = st.selectbox("Search Destination", list(LANDMARKS.keys()), key="map_landmark")
            
        with col2:
            sel_hour = st.slider("Hour of day", 7, 22, 14, format="%d:00", key="map_hour_global")
            current_month = datetime.datetime.now().month
            sel_month = st.selectbox("Month", list(MONTH_OPTIONS.keys()), format_func=lambda x: MONTH_OPTIONS[x], index=current_month-1, key="map_month_global")
            sel_time_limit = st.slider("Minimum Time Limit (mins)", 30, 240, 30, step=30, key="map_time_limit")
            
        with col3:
            sel_weather = st.selectbox("Weather conditions", list(WEATHER_OPTIONS.keys()), key="map_weather_global")
            sel_holiday = st.checkbox("Public holiday", key="map_holiday_global")
            enable_anim = st.checkbox("Enable Daily Animation (ignores 'Hour' slider)", key="map_anim")
            color_by_price = st.checkbox("Color Map by Hourly Rate ($/hr)", key="map_pricing")

    sel_temp, sel_precip = WEATHER_OPTIONS[sel_weather]
    dow_val = DAYS.index(sel_day)
    bf_loc = st.session_state["bf_loc"].copy()
    
    # Optional Filtering by Time Limit and Landmark Distance
    if "time_limit" in bf_loc.columns:
        bf_loc = bf_loc[bf_loc["time_limit"].fillna(0) >= sel_time_limit]
        
    if sel_landmark != "None":
        l_lat, l_lon = LANDMARKS[sel_landmark]
        dist = np.sqrt((bf_loc["lat"] - l_lat)**2 + (bf_loc["lon"] - l_lon)**2)
        bf_loc = bf_loc[dist < 0.006]
        if bf_loc.empty:
            st.warning(f"No parking blocks found within walking distance of {sel_landmark} with current filters.")
            return

    global_mean = float(train_df["occ_rate"].mean())
    enc = st.session_state.get("block_encoder")
    bf_mean_tbl      = st.session_state.get("bf_mean", pd.DataFrame())
    bf_hour_mean_tbl = st.session_state.get("bf_hour_mean", pd.DataFrame())
    bf_dow_mean_tbl  = st.session_state.get("bf_dow_mean", pd.DataFrame())

    # ── 2. Vectorized Feature Matrix Builder ─────────────────────
    def build_batch(target_hour, base_loc):
        m_df = base_loc[["block", "lon", "lat"]].copy()
        
        if enc:
            blocks_to_encode = m_df["block"].astype(str)
            known_classes = set(enc.classes_)
            safe_blocks = [b if b in known_classes else enc.classes_[0] for b in blocks_to_encode]
            m_df["block_id"] = enc.transform(safe_blocks)
        else:
            m_df["block_id"] = 0

        m_df = m_df.merge(bf_mean_tbl[["block", "blockface_mean"]], on="block", how="left")
        
        hm_cols = ["block", "blockface_hour_mean"]
        if "rate" in bf_hour_mean_tbl.columns:
            hm_cols.append("rate")
            
        hm_df = bf_hour_mean_tbl[bf_hour_mean_tbl["hour"] == target_hour][hm_cols] if not bf_hour_mean_tbl.empty else pd.DataFrame(columns=hm_cols)
        dm_df = bf_dow_mean_tbl[bf_dow_mean_tbl["dow"] == dow_val][["block", "blockface_dow_mean"]] if not bf_dow_mean_tbl.empty else pd.DataFrame(columns=["block", "blockface_dow_mean"])
        
        m_df = m_df.merge(hm_df, on="block", how="left")
        m_df = m_df.merge(dm_df, on="block", how="left")
        
        m_df["blockface_mean"] = m_df["blockface_mean"].fillna(global_mean)
        m_df["blockface_hour_mean"] = m_df["blockface_hour_mean"].fillna(global_mean)
        m_df["blockface_dow_mean"] = m_df["blockface_dow_mean"].fillna(global_mean)
        
        if "rate" in m_df.columns:
            m_df["rate"] = m_df["rate"].fillna(0)

        m_df["hour"] = target_hour
        m_df["dow"]  = dow_val
        m_df["month"] = sel_month
        m_df["is_weekend"] = int(dow_val >= 5)
        m_df["is_lunch"]   = int(11 <= target_hour <= 13)
        m_df["is_evening"] = int(17 <= target_hour <= 19)
        m_df["is_fri_pm"]  = int(dow_val == 4 and 14 <= target_hour <= 18)
        m_df["hour_sin"]   = np.sin(2 * np.pi * target_hour / 24)
        m_df["hour_cos"]   = np.cos(2 * np.pi * target_hour / 24)
        m_df["dow_sin"]    = np.sin(2 * np.pi * dow_val / 7)
        m_df["dow_cos"]    = np.cos(2 * np.pi * dow_val / 7)
        m_df["is_holiday"] = int(sel_holiday)
        m_df["is_holiday_eve"] = 0
        m_df["temp_c"]     = sel_temp
        m_df["precip_mm"]  = sel_precip
        m_df["is_rainy"]   = int(sel_precip > 0.5)

        X_batch = m_df[FEATURES]
        preds = regressors[sel_model].predict(X_batch).clip(0, 1)
        m_df["predicted_occ"] = preds

        def get_cat(rate):
            if rate < 0.4: return "Low"
            elif rate < 0.7: return "Medium"
            else: return "High"
            
        m_df["predicted_cat"] = m_df["predicted_occ"].apply(get_cat)
        m_df["hour_str"] = f"{target_hour:02d}:00"
        return m_df

    # ── 3. Handle Animation vs Single Hour ───────────────────────
    with st.spinner("Generating precision feature vectors..."):
        if enable_anim:
            frames = []
            for h in range(7, 23):
                frames.append(build_batch(h, bf_loc))
            final_map_df = pd.concat(frames, ignore_index=True)
            anim_kw = {"animation_frame": "hour_str"}
        else:
            final_map_df = build_batch(sel_hour, bf_loc)
            anim_kw = {}

    # ── 4. Render Map ───────────────────────────────────────────
    if sel_landmark != "None":
        st.markdown(f"### Easiest Parking near {sel_landmark}")
    else:
        st.markdown("### Interactive Prediction Map")

    color_col = "rate" if color_by_price and "rate" in final_map_df.columns else "predicted_occ"

    c_scale = "Blues" if color_col == "rate" else ["#00cc96", "#ffa15a", "#ef553b"]
    c_range = None if color_col == "rate" else [0, 1]
    hover_data = {"predicted_occ": ":.0%", "lat": False, "lon": False}
    if "rate" in final_map_df.columns:
        hover_data["rate"] = ":.2f"

    fig = px.scatter_mapbox(
        final_map_df, lat="lat", lon="lon", hover_name="block", custom_data=["block"],
        hover_data=hover_data, color=color_col, color_continuous_scale=c_scale,
        range_color=c_range, size_max=15, zoom=14.5 if sel_landmark == "None" else 15.5, 
        opacity=0.8, template="plotly_dark",
        labels={"predicted_occ": "Predicted Occ", "rate": "Price $/hr", "hour_str": "Time", "customdata[0]": "Block"},
        **anim_kw
    )
    fig.update_layout(mapbox_style="carto-darkmatter")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(clickmode="event+select")

    # If animation is active, Plotly selection conflicts with the slider. So we only enable selection if not animating.
    if enable_anim:
        st.plotly_chart(fig, use_container_width=True)
        st.info("Animation mode is active. Turn off 'Enable Daily Animation' to click on individual blocks.")
        event = None
    else:
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="plotly_map")

    # ── 5. Destination Routing Recommendations ──────────────────
    if sel_landmark != "None" and not enable_anim:
        st.markdown(f"**Top 3 Easiest Parking blocks near {sel_landmark} at {sel_hour}:00**")
        top3 = final_map_df.sort_values("predicted_occ").head(3)
        for _, row in top3.iterrows():
            rate_str = f" - ${row['rate']:.2f}/hr" if "rate" in row else ""
            st.write(f"- `{row['block']}` (Predicted Occupancy: **{row['predicted_occ']:.0%}**{rate_str})")
        st.markdown("---")

    sel_block_fallback = st.selectbox(
        "🔍 Pick a Block manually to see detailed predictions (Fallback if map click is unresponsive):", 
        ["None"] + final_map_df["block"].tolist()
    )

    # If they manually select from the dropdown, override the map click
    if sel_block_fallback != "None":
        st.session_state["sel_map_block"] = sel_block_fallback

    try:
        # Reverting back to the original correctly functioning attribute check
        if event and hasattr(event, "selection") and event.selection:
            pts = event.selection.get("points", []) if hasattr(event.selection, "get") else getattr(event.selection, "points", [])
            
            if pts and len(pts) > 0:
                pt = pts[0]
                pt_dict = pt if isinstance(pt, dict) else pt.__dict__
                
                # Forcefully extract the block ID from customdata payload
                customdata = pt_dict.get("customdata", [])
                if customdata and len(customdata) > 0:
                    st.session_state["sel_map_block"] = customdata[0]
                else:
                    # Fallback to row index mapping
                    idx = pt_dict.get("point_index", pt_dict.get("pointIndex", -1))
                    if idx is not None and 0 <= int(idx) < len(final_map_df):
                        st.session_state["sel_map_block"] = final_map_df.iloc[int(idx)]["block"]
    except Exception as e:
        pass

    sel_block = st.session_state.get("sel_map_block")

    if not sel_block:
        st.info("👆 Click on any hotspot on the map to see the prediction details for that block.")
        return

    # ── 6. Output Panel for Selected Block ──────────────────────
    st.markdown(f"**Selected Block:** `{sel_block}`")
    st.markdown("---")

    selected_row = final_map_df[final_map_df["block"] == sel_block].iloc[0]
    pred_rate = selected_row["predicted_occ"]
    pred_cat  = selected_row["predicted_cat"]
    bf_mean   = selected_row["blockface_mean"]
    
    color = CAT_COLOR.get(pred_cat, "blue")
    margin = min(abs(pred_rate - 0.4), abs(pred_rate - 0.7))
    confidence = "high" if margin > 0.1 else "moderate"

    r2 = 0
    if sel_model in reg_results:
        r2 = reg_results[sel_model]["R²"]

    col_out1, col_out2 = st.columns([1, 1], gap="large")

    with col_out1:
        st.subheader("Prediction result")
        m1, m2 = st.columns(2)
        m1.metric("Predicted occupancy", f"{pred_rate:.0%}")
        m2.metric("Model R²", f"{r2}")

        st.markdown(f"**Category:** :{color}[**{pred_cat} occupancy**] (confidence: {confidence})")
        if "rate" in selected_row and pd.notna(selected_row["rate"]):
            st.markdown(f"**Hourly Rate:** ${selected_row['rate']:.2f}/hr")

        if "block" in train_df.columns and sel_block in train_df["block"].values:
            mode_vals = train_df[train_df["block"] == sel_block]["capacity"].mode()
            capacity  = int(mode_vals.iloc[0]) if not mode_vals.empty else int(train_df["capacity"].mode().iloc[0])
        else:
            capacity = int(train_df["capacity"].mode().iloc[0])

        available = max(0, round(capacity * (1 - pred_rate)))
        st.info(f"**Estimated available spaces:** {available} / {capacity} — based on historical avg capacity for this block")

    with col_out2:
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

        if sel_model in reg_results:
            st.caption(f"Model: {sel_model} | MAE: {reg_results[sel_model].get('MAE', '')} | Historical avg for this block: {bf_mean:.0%}")
