# modules/model_utils.py
import os
import uuid
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, Optional

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score,
    r2_score, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Optional third-party models
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------------------------------------
#  TASK TYPE DETECTION
# -----------------------------------------------------------
def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Auto-detect classification vs regression."""
    col = df[target_col]

    if col.dtype == object:
        return "classification"

    if col.nunique() <= 15:
        return "classification"

    return "regression"


# -----------------------------------------------------------
#  QUICK MODE MODELS — SVM REMOVED
# -----------------------------------------------------------
def get_quick_models(task: str) -> Dict[str, Any]:
    """Ultra-fast Quick Mode models (ALWAYS < 30 seconds)."""
    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1500),
            "Random Forest (fast)": RandomForestClassifier(
                random_state=42,
                n_estimators=50,
                max_depth=12
            ),
        }
        if LIGHTGBM_AVAILABLE:
            models["LightGBM (fast)"] = LGBMClassifier(
                random_state=42,
                n_estimators=40,
                num_leaves=31
            )
        return models

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor (fast)": RandomForestRegressor(
                random_state=42,
                n_estimators=50,
                max_depth=12
            ),
        }
        if LIGHTGBM_AVAILABLE:
            models["LightGBM Regressor (fast)"] = LGBMRegressor(
                random_state=42,
                n_estimators=40,
                num_leaves=31
            )
        return models


# -----------------------------------------------------------
#  FULL MODEL LIST (NO SVM)
# -----------------------------------------------------------
def _get_candidate_models(task: str) -> Dict[str, Any]:
    """All available models except SVM."""
    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
            "KNN": KNeighborsClassifier(),
            "Gaussian NB": GaussianNB()
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False)
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMClassifier(random_state=42)
        if CATBOOST_AVAILABLE:
            models["CatBoost"] = CatBoostClassifier(verbose=0, random_seed=42)

        return models

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=200),
        }
        if XGBOOST_AVAILABLE:
            models["XGBoost Regressor"] = XGBRegressor(random_state=42)
        if LIGHTGBM_AVAILABLE:
            models["LightGBM Regressor"] = LGBMRegressor(random_state=42)
        if CATBOOST_AVAILABLE:
            models["CatBoost Regressor"] = CatBoostRegressor(verbose=0, random_seed=42)

        return models


# -----------------------------------------------------------
#  OPTIONAL HYPERPARAMETER TUNING (NO SVM)
# -----------------------------------------------------------
def _get_tuning_grid(name: str, task: str) -> dict:
    """Lightweight tuning grid."""
    if task == "classification":
        grids = {
            "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 12]},
            "Logistic Regression": {"C": [0.1, 1.0]},
            "KNN": {"n_neighbors": [3, 5]},
            "XGBoost": {"n_estimators": [50, 100], "max_depth": [3, 6]},
            "LightGBM": {"n_estimators": [50, 100]},
            "CatBoost": {"iterations": [100], "depth": [4]}
        }
    else:
        grids = {
            "Random Forest Regressor": {"n_estimators": [100, 200]},
            "XGBoost Regressor": {"n_estimators": [50, 100]},
            "LightGBM Regressor": {"n_estimators": [50, 100]},
            "CatBoost Regressor": {"iterations": [100], "depth": [4]}
        }
    return grids.get(name, {})


# -----------------------------------------------------------
#  TRAINING PIPELINE
# -----------------------------------------------------------
def train_models(
    df: pd.DataFrame,
    target_col: str,
    tune: bool = False,
    mode: str = "full",
    cv: int = 3,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:

    try:
        df = df.copy().dropna(subset=[target_col])

        # detect classification or regression
        task = detect_task_type(df, target_col)

        # encode target if classification
        y_raw = df[target_col]
        target_mapping = None

        if task == "classification":
            if y_raw.dtype == object:
                unique_vals = sorted([str(v) for v in y_raw.unique()])
                target_mapping = {unique_vals[i]: i for i in range(len(unique_vals))}
                y = y_raw.astype(str).map(target_mapping)
            else:
                y = y_raw
        else:
            y = pd.to_numeric(y_raw)

        X = df.drop(columns=[target_col])

        # simple encoding: convert all non-numerics to numeric codes
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype(str).factorize()[0]

        stratify = y if task == "classification" else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        # choose models
        if mode == "quick":
            candidates = get_quick_models(task)
        else:
            candidates = _get_candidate_models(task)

        total = len(candidates)
        results = []
        best_score = -np.inf
        best_model_obj = None
        best_model_name = None

        for idx, (name, model) in enumerate(candidates.items()):
            # progress
            if progress_callback:
                pct = int((idx / total) * 100)
                progress_callback(pct, f"Training {name} ({idx+1}/{total})...")

            try:
                est = model

                if tune:
                    grid = _get_tuning_grid(name, task)
                    if grid:
                        gs = GridSearchCV(est, grid, cv=cv, n_jobs=-1)
                        gs.fit(X_train, y_train)
                        est = gs.best_estimator_

                # train model
                est.fit(X_train, y_train)
                preds = est.predict(X_test)

                # evaluate
                if task == "classification":
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="weighted")
                    score = (acc + f1) / 2
                    results.append({
                        "Model": name,
                        "Accuracy": acc,
                        "F1 Score": f1,
                        "Tuned": tune
                    })
                else:
                    r2 = r2_score(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    score = r2
                    results.append({
                        "Model": name,
                        "R2 Score": r2,
                        "RMSE": rmse,
                        "Tuned": tune
                    })

                if score > best_score:
                    best_score = score
                    best_model_obj = est
                    best_model_name = name

            except Exception as e:
                results.append({"Model": name, "Error": str(e)})

        # save best model
        model_path = os.path.join(
            MODEL_DIR,
            f"{uuid.uuid4().hex[:8]}_{best_model_name.replace(' ', '_')}.pkl"
        )
        pickle.dump(best_model_obj, open(model_path, "wb"))

        if progress_callback:
            progress_callback(100, f"Finished! Best Model → {best_model_name}")

        return {
            "task": task,
            "results": results,
            "best_model": best_model_name,
            "model_path": model_path
        }

    except Exception as e:
        return {"error": str(e)}
