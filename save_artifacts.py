"""
save_artifacts.py
Run this ONCE locally to pre-compute and save all artifacts.
These files are then pushed to GitHub and loaded by app.py on Streamlit Cloud.

Run:
    python save_artifacts.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

from data_loader import load_and_clean, FEATURES
from model import train_models, evaluate

# ── Step 1: Load and clean training data ─────────────────────
print("Loading training data...")
train_df, lookup = load_and_clean("belltown_2023_full.csv")
print(f"  train_df shape: {train_df.shape}")

# ── Step 2: Train models ──────────────────────────────────────
print("Training models...")
regressors, feat_imp = train_models(train_df)
print("  Done training.")

# ── Step 3: Save models and feature importance ────────────────
print("Saving models.joblib and feat_imp.joblib...")
joblib.dump(regressors, "models.joblib")
joblib.dump(feat_imp,   "feat_imp.joblib")

# ── Step 4: Save lookup tables ────────────────────────────────
print("Saving lookup CSVs...")
lookup["bf_mean"].to_csv("bf_mean.csv",           index=False)
lookup["bf_hour_mean"].to_csv("bf_hour_mean.csv", index=False)
lookup["bf_dow_mean"].to_csv("bf_dow_mean.csv",   index=False)
lookup["bf_loc"].to_csv("bf_loc.csv",             index=False)

# ── Step 5: Save block encoder ────────────────────────────────
import streamlit as st
print("Saving block_encoder.joblib...")
import os; os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# encoder is stored in session_state during load_and_clean
# re-extract it
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_df["block"].astype(str))
joblib.dump(le, "block_encoder.joblib")

# ── Step 6: Compute and save eval results ────────────────────
print("Computing eval results on holdout...")
holdout_df = train_df.sample(frac=0.2, random_state=42)

eval_result = evaluate(regressors, holdout_df)

X_hold = holdout_df[FEATURES]
y_hold = holdout_df["occ_rate"]
holdout_results = {}
for name, m in regressors.items():
    pred = m.predict(X_hold).clip(0, 1)
    holdout_results[name] = {
        "R²":  round(r2_score(y_hold, pred), 3),
        "MAE": round(mean_absolute_error(y_hold, pred), 4),
    }

# ── Evaluate on last 30 days (external test) ─────────────────
print("Evaluating on last 30 days test set...")
load_and_clean.clear()  
evaluate.clear()
test_df, _ = load_and_clean("belltown_last30days.csv")
eval_external = evaluate(regressors, test_df)

joblib.dump({
    "reg_results":     eval_external["reg_results"],
    "cm":              eval_external["cm"],
    "clf_acc":         eval_external["clf_acc"],
    "scatter_df":      eval_external["scatter_df"],
    "holdout_results": holdout_results,
}, "eval_results.joblib")
print("  Saved eval_results.joblib (external + holdout)")

# ── Step 7: Save train sample for EDA ────────────────────────
print("Saving train_sample.csv...")
train_df.sample(frac=0.1, random_state=42).to_csv("train_sample.csv", index=False)

print("\nAll done! Files saved:")
import subprocess
subprocess.run(["ls", "-lh",
    "models.joblib", "feat_imp.joblib", "block_encoder.joblib",
    "eval_results.joblib", "bf_mean.csv", "bf_hour_mean.csv",
    "bf_dow_mean.csv", "bf_loc.csv", "train_sample.csv"
])