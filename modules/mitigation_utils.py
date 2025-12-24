# modules/mitigation_utils.py
import pandas as pd
import os, uuid
import pickle
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Dict, Any
from .serialization_utils import save_model_wrapper

REPAIRED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "repaired")
os.makedirs(REPAIRED_DIR, exist_ok=True)

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

        base = LogisticRegression(max_iter=1000)
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
