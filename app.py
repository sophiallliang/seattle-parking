"""
app.py
Entry point — loads pre-saved artifacts instead of training from scratch.

Run:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import streamlit as st
import pandas as pd
import gdown

import _pages.overview        as pg_overview
import _pages.predict         as pg_predict
import _pages.explore         as pg_explore
import _pages.model_eval      as pg_model_eval
import _pages.interactive_map as pg_map

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Belltown Parking",
    page_icon="🅿",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("🅿 Belltown Parking")
    st.markdown("Occupancy Prediction System")
    st.markdown("---")
    st.caption("Project: DS capstone · Belltown")

# ── Download models.joblib from Google Drive if missing ──────
MODELS_GDRIVE_URL = "https://drive.google.com/file/d/1ifvvbXqQ9TycQkCTeZ28DpxBKJ1TRM1q/view?usp=drive_link"  

if not os.path.exists("models.joblib"):
    with st.spinner("Downloading models... (first run only, ~1 min)"):
        gdown.download(MODELS_GDRIVE_URL, "models.joblib", fuzzy=True, quiet=False)

# ── Load all artifacts ───────────────────────────────────────
with st.spinner("Loading models and data..."):
    regressors   = joblib.load("models.joblib")
    feat_imp     = joblib.load("feat_imp.joblib")
    block_encoder = joblib.load("block_encoder.joblib")
    eval_cache   = joblib.load("eval_results.joblib")

    bf_mean      = pd.read_csv("bf_mean.csv")
    bf_hour_mean = pd.read_csv("bf_hour_mean.csv")
    bf_dow_mean  = pd.read_csv("bf_dow_mean.csv")
    bf_loc       = pd.read_csv("bf_loc.csv")
    train_df     = pd.read_csv("train_sample.csv")

# ── Restore session_state ────────────────────────────────────
st.session_state["bf_mean"]        = bf_mean
st.session_state["bf_hour_mean"]   = bf_hour_mean
st.session_state["bf_dow_mean"]    = bf_dow_mean
st.session_state["bf_loc"]         = bf_loc
st.session_state["block_encoder"]  = block_encoder
st.session_state["block_names"]    = list(block_encoder.classes_)

# ── Unpack eval results ──────────────────────────────────────
reg_results     = eval_cache["reg_results"]
cm              = eval_cache["cm"]
clf_acc         = eval_cache["clf_acc"]
scatter_df      = eval_cache["scatter_df"]
holdout_results = eval_cache["holdout_results"]
is_holdout      = True

# ── Tab routing ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Map",
    "Overview",
    "Predict",
    "Explore",
    "Model",
])

with tab1:
    pg_map.render(regressors, None, reg_results, train_df)

with tab2:
    pg_overview.render(train_df, train_df)

with tab3:
    pg_predict.render(regressors, reg_results, train_df)

with tab4:
    pg_explore.render(train_df)

with tab5:
    pg_model_eval.render(
        reg_results, cm, clf_acc,
        feat_imp, scatter_df,
        is_holdout,
        holdout_results=holdout_results,
    )