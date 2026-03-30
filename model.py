"""
model.py
Model training, evaluation, and prediction logic.

"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, accuracy_score,
)
import xgboost as xgb

from data_loader import FEATURES, OCC_LABELS

# ── Model registry ───────────────────────────────────────────
MODEL_NAMES = ["XGBoost", "Random Forest", "Linear Regression"]


def _build_regressors() -> dict:
    return {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=800, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, gamma=0.1,
            random_state=42, verbosity=0,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=15,
            min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
        "Linear Regression": LinearRegression(),
    }


# ── Training ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(_train_df: pd.DataFrame):
    """
    Train all regressors on training data.
    Returns (regressors dict, feature_importance Series).

    Prefixing _train_df with _ tells Streamlit not to hash the DataFrame,
    avoiding re-training on every minor state change.
    """
    X     = _train_df[FEATURES]
    y_reg = _train_df["occ_rate"]

    regressors = _build_regressors()
    for m in regressors.values():
        m.fit(X, y_reg)

    feat_imp = pd.Series(
        regressors["Random Forest"].feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)

    return regressors, feat_imp


# ── Evaluation ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def evaluate(_regressors, _test_df: pd.DataFrame) -> dict:
    """
    Evaluate all models on the test set.
    Returns a dict with regression metrics, confusion matrix, and accuracy.
    """
    X_test = _test_df[FEATURES]
    y_reg  = _test_df["occ_rate"]
    y_clf  = _test_df["occ_cat"]

    reg_results = {}
    for name, m in _regressors.items():
        pred = m.predict(X_test).clip(0, 1)
        reg_results[name] = {
            "R²":   round(r2_score(y_reg, pred), 3),
            "MAE":  round(mean_absolute_error(y_reg, pred), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_reg, pred))), 4),
        }

    # Predicted vs actual for scatter plot (Random Forest)
    rf_pred = _regressors["Random Forest"].predict(X_test).clip(0, 1)
    scatter_df = pd.DataFrame({
        "Actual":    y_reg.values,
        "Predicted": rf_pred,
    })

    # Classification using RF predictions + fixed thresholds
    rf_cat = pd.cut(rf_pred, bins=[-0.01, 0.4, 0.7, 1.01], labels=OCC_LABELS)
    cm  = confusion_matrix(y_clf, rf_cat, labels=OCC_LABELS)
    acc = round(accuracy_score(y_clf, rf_cat), 3)

    return {
        "reg_results": reg_results,
        "cm":          cm,
        "clf_acc":     acc,
        "scatter_df":  scatter_df,
    }


# ── Prediction ───────────────────────────────────────────────
def predict(regressors: dict, model_name: str,
            input_row: pd.DataFrame) -> dict:
    """
    Run a single prediction.
    Category is derived directly from rate using fixed thresholds
    to ensure consistency between numeric value and label.
    """
    rate = float(regressors[model_name].predict(input_row).clip(0, 1)[0])

    if rate < 0.4:
        cat = "Low"
    elif rate < 0.7:
        cat = "Medium"
    else:
        cat = "High"

    margin = min(abs(rate - 0.4), abs(rate - 0.7))
    confidence = "high" if margin > 0.1 else "moderate"

    return {
        "rate":       rate,
        "category":   cat,
        "confidence": confidence,
    }


# ── Model quality helper ─────────────────────────────────────
def quality_label(r2: float) -> tuple[str, str]:
    """Returns (label, color) based on R² value."""
    if r2 >= 0.80:
        return "Excellent", "green"
    elif r2 >= 0.70:
        return "Good", "blue"
    elif r2 >= 0.50:
        return "Fair", "orange"
    else:
        return "Poor — needs improvement", "red"