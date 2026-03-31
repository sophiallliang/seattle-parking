"""
app.py
Entry point — loads CSV directly, trains models, tab routing.

Run:
    streamlit run app.py
"""

import os
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

from data_loader import load_and_clean, FEATURES
from model import train_models, evaluate

import _pages.overview   as pg_overview
import _pages.predict    as pg_predict
import _pages.explore    as pg_explore
import _pages.model_eval as pg_model_eval
import _pages.interactive_map as pg_map

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Belltown Parking",
    page_icon="🅿",
    layout="wide",
)

# ── Data paths ───────────────────────────────────────────────
TRAIN_PATH = "belltown_2023_full.csv"
TEST_PATH  = "belltown_last30days.csv"

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("🅿 Belltown Parking")
    st.markdown("Occupancy Prediction System")
    st.markdown("---")
    st.caption("Project: DS capstone · Belltown")

# ── Load training data ───────────────────────────────────────
with st.spinner("Loading and cleaning training data…"):
    train_df, lookup = load_and_clean(TRAIN_PATH)

st.session_state["bf_mean"] = lookup.get("bf_mean")
st.session_state["bf_hour_mean"] = lookup.get("bf_hour_mean")
st.session_state["bf_dow_mean"] = lookup.get("bf_dow_mean")
if "bf_loc" in lookup:
    st.session_state["bf_loc"] = lookup.get("bf_loc")

# ── Fix: session_state is empty on cache hit, force re-run ───
if "block_encoder" not in st.session_state:
    load_and_clean.clear()
    train_df = load_and_clean(TRAIN_PATH)

if len(train_df) == 0:
    st.error(
        "No Belltown records found after filtering. "
        "Check that the CSV contains Belltown parking data."
    )
    st.stop()

# ── Holdout (always 20% of train) ────────────────────────────
holdout_df = train_df.sample(frac=0.2, random_state=42)

# ── Load test data ───────────────────────────────────────────
is_holdout = True
test_df    = holdout_df

if os.path.exists(TEST_PATH):
    with st.spinner("Loading last-30-days test data…"):
        test_df, _ = load_and_clean(TEST_PATH)
        is_holdout = False
    st.sidebar.success(f"Test data: {len(test_df):,} records")
else:
    st.sidebar.info("Using 20% holdout as test set")

# ── Train models ─────────────────────────────────────────────
with st.spinner("Training models… (cached after first run)"):
    regressors, feat_imp = train_models(train_df)

# ── Evaluate on test set (last 30 days or holdout) ───────────
eval_result = evaluate(regressors, test_df)
reg_results = eval_result["reg_results"]
cm          = eval_result["cm"]
clf_acc     = eval_result["clf_acc"]
scatter_df  = eval_result["scatter_df"]

# ── Evaluate on holdout (always computed directly, no cache) ──
X_hold = holdout_df[FEATURES]
y_hold = holdout_df["occ_rate"]
holdout_results = {}
for name, m in regressors.items():
    pred = m.predict(X_hold).clip(0, 1)
    holdout_results[name] = {
        "R²":  round(r2_score(y_hold, pred), 3),
        "MAE": round(mean_absolute_error(y_hold, pred), 4),
    }

# ── Tab routing ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Map",
    "Overview",
    "Predict",
    "Explore",
    "Model",
])

with tab1:
    pg_map.render(regressors, clf, reg_results, train_df)

with tab2:
    pg_overview.render(train_df, test_df)

with tab3:
    pg_predict.render(regressors, clf, reg_results, train_df)

with tab4:
    pg_explore.render(train_df)

with tab5:
    pg_model_eval.render(
        reg_results, cm, clf_acc,
        feat_imp, scatter_df,
        is_holdout,
        holdout_results=holdout_results,
    )