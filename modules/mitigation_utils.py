# modules/mitigation_utils.py
import pandas as pd
import os, uuid
import pickle
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, GridSearch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Dict, Any
from .serialization_utils import save_model_wrapper

REPAIRED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "repaired")
os.makedirs(REPAIRED_DIR, exist_ok=True)


def _get_best_base_estimator():
    """
    FIX 4: Returns the best available base estimator for mitigation.
    Tries LightGBM first (matches paper claims), then RandomForest, then falls back to
    LogisticRegression. All Fairlearn reducers work with any sklearn-compatible estimator.
    """
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, num_leaves=31, random_state=42, verbose=-1)
    except ImportError:
        pass
    try:
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    except Exception:
        pass
    return LogisticRegression(max_iter=1000)


def cleanup_repaired_dir(keep_newest: int = 5) -> int:
    """
    FIX 5: Deletes old GridSearch .pkl files from the repaired directory.
    Keeps only the `keep_newest` most-recent files to prevent disk bloat.
    Returns the number of files deleted.
    """
    try:
        files = [
            os.path.join(REPAIRED_DIR, f)
            for f in os.listdir(REPAIRED_DIR)
            if f.endswith(".pkl")
        ]
        files.sort(key=os.path.getmtime, reverse=True)  # newest first
        files_to_delete = files[keep_newest:]
        for fp in files_to_delete:
            try:
                os.remove(fp)
            except Exception:
                pass
        return len(files_to_delete)
    except Exception:
        return 0


def _build_feature_maps(X: pd.DataFrame):
    feature_maps = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            feature_maps[col] = None
        else:
            vals, uniques = pd.factorize(X[col].astype(str))
            mapping = {str(v): int(i) for i, v in enumerate(uniques)}
            feature_maps[col] = mapping
    return feature_maps

def _apply_feature_maps(X: pd.DataFrame, feature_maps: dict):
    X_enc = X.copy()
    for col, fmap in feature_maps.items():
        if col not in X_enc.columns:
            X_enc[col] = -1
            continue
        if fmap is None:
            try:
                X_enc[col] = pd.to_numeric(X_enc[col])
            except Exception:
                X_enc[col] = pd.factorize(X_enc[col].astype(str))[0]
        else:
            X_enc[col] = X_enc[col].astype(str).map(fmap).fillna(-1).astype(int)
    return X_enc

def mitigate_bias_reweighing(df: pd.DataFrame, target_col: str, sensitive_col: str) -> Dict[str, Any]:
    try:
        dfc = df.copy().dropna(subset=[target_col, sensitive_col]).reset_index(drop=True)
        unique_vals = list(pd.Series(dfc[target_col].unique()))
        if len(unique_vals) != 2:
            return {"status": "error", "message": f"Target must be binary. Found {len(unique_vals)} classes: {unique_vals}"}

        u_sorted = sorted([str(v) for v in unique_vals])
        target_mapping = {u_sorted[0]: 0, u_sorted[1]: 1}
        dfc[target_col] = dfc[target_col].astype(str).map(target_mapping)

        X_raw = dfc.drop(columns=[target_col])
        feature_maps = _build_feature_maps(X_raw)
        X_enc = _apply_feature_maps(X_raw, feature_maps)
        y = dfc[target_col].astype(int)
        sens = dfc[sensitive_col].astype(str)

        stratify = y if len(y.unique()) > 1 else None
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X_enc, y, sens, test_size=0.3, random_state=42, stratify=stratify
        )

        base = _get_best_base_estimator()  # FIX 4: LightGBM > RF > LR, not always just LR
        constraint = DemographicParity()
        mitigator = ExponentiatedGradient(base, constraints=constraint)
        mitigator.fit(X_train, y_train, sensitive_features=sens_train)

        wrapper = {
            "model": mitigator,
            "feature_maps": feature_maps,
            "feature_cols": list(X_enc.columns),
            "target_mapping": target_mapping,
            "task": "classification"
        }
        model_path = os.path.join(REPAIRED_DIR, f"{uuid.uuid4().hex[:8]}_mitigated.pkl")
        save_model_wrapper(wrapper, model_path)

        y_pred = mitigator.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))

        return {"status": "success", "model_path": model_path, "accuracy": acc, "label_mapping": target_mapping, "message": "Mitigation done"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def mitigate_bias_grid_search(df: pd.DataFrame, target_col: str, sensitive_col: str, grid_size: int = 10) -> Dict[str, Any]:
    try:
        dfc = df.copy().dropna(subset=[target_col, sensitive_col]).reset_index(drop=True)
        unique_vals = list(pd.Series(dfc[target_col].unique()))
        if len(unique_vals) != 2:
            return {"status": "error", "message": f"Target must be binary. Found {len(unique_vals)} classes."}

        u_sorted = sorted([str(v) for v in unique_vals])
        target_mapping = {u_sorted[0]: 0, u_sorted[1]: 1}
        dfc[target_col] = dfc[target_col].astype(str).map(target_mapping)

        X_raw = dfc.drop(columns=[target_col])
        feature_maps = _build_feature_maps(X_raw)
        X_enc = _apply_feature_maps(X_raw, feature_maps)
        y = dfc[target_col].astype(int)
        sens = dfc[sensitive_col].astype(str)

        stratify = y if len(y.unique()) > 1 else None
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X_enc, y, sens, test_size=0.3, random_state=42, stratify=stratify
        )

        base = _get_best_base_estimator()  # FIX 4: LightGBM > RF > LR
        constraint = DemographicParity()
        sweep = GridSearch(base, constraints=constraint, grid_size=grid_size)
        sweep.fit(X_train, y_train, sensitive_features=sens_train)

        models_data = []
        for i, predictor in enumerate(sweep.predictors_):
            y_pred = predictor.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            
            # Approximation of fairness for the UI tradeoff curve
            # We use the UI to calculate real fairness, but we need basic scores here
            from .fairness_metrics import compute_model_fairness
            fairness_res = compute_model_fairness(y_test, y_pred, sens_test)
            dp_diff = fairness_res.get("Demographic Parity Difference", 1.0)
            
            wrapper = {
                "model": predictor,
                "feature_maps": feature_maps,
                "feature_cols": list(X_enc.columns),
                "target_mapping": target_mapping,
                "task": "classification"
            }
            # Save each candidate model uniquely
            model_path = os.path.join(REPAIRED_DIR, f"{uuid.uuid4().hex[:8]}_grid_{i}.pkl")
            save_model_wrapper(wrapper, model_path)
            
            models_data.append({
                "model_id": i,
                "accuracy": acc,
                "dp_diff": dp_diff,
                "model_path": model_path,
                "wrapper": wrapper
            })

        return {"status": "success", "models": models_data, "label_mapping": target_mapping}
    except Exception as e:
        return {"status": "error", "message": str(e)}
