# modules/fairness_metrics.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score
from typing import Any, Dict

def _encode_binary(y: pd.Series):
    y_s = pd.Series(y).reset_index(drop=True)
    if pd.api.types.is_integer_dtype(y_s) or pd.api.types.is_float_dtype(y_s):
        unique_vals = sorted(y_s.dropna().unique())
        if set(unique_vals).issubset({0, 1}):
            return y_s.fillna(0).astype(int).to_numpy(), ['0','1']
    le = LabelEncoder()
    try:
        y_enc = le.fit_transform(y_s.astype(str))
        classes = list(le.classes_)
        if len(classes) == 2:
            return y_enc, classes
        else:
            pos = y_s.mode().iloc[0]
            y_bin = (y_s.astype(str) == str(pos)).astype(int).to_numpy()
            return y_bin, [f"not_{pos}", str(pos)]
    except Exception:
        y_bin = (~y_s.isnull()).astype(int).to_numpy()
        return y_bin, ['not_null', 'present']

def compute_model_fairness(y_true, y_pred, sensitive_features) -> Dict[str, Any]:
    try:
        y_true_s = pd.Series(y_true).reset_index(drop=True)
        y_pred_s = pd.Series(y_pred).reset_index(drop=True)
        sens_s = pd.Series(sensitive_features).reset_index(drop=True)

        if not (len(y_true_s) == len(y_pred_s) == len(sens_s)):
            return {"error": "Length mismatch between y_true, y_pred and sensitive feature."}

        y_true_enc, classes = _encode_binary(y_true_s)
        y_pred_enc, _ = _encode_binary(y_pred_s)

        unique = np.unique(y_true_enc)
        if not set(unique).issubset({0,1}):
            return {"error": "Supplied y labels are not binary after encoding."}

        groups = {}
        for g in sens_s.dropna().unique():
            mask = sens_s == g
            if mask.sum() == 0:
                continue
            acc = float(accuracy_score(y_true_enc[mask], y_pred_enc[mask]))
            rec = float(recall_score(y_true_enc[mask], y_pred_enc[mask], zero_division=0))
            sel_rate = float((y_pred_enc[mask] == 1).mean()) if mask.sum() > 0 else 0.0
            groups[str(g)] = {"accuracy": acc, "recall": rec, "selection_rate": sel_rate}

        if len(groups) == 0:
            return {"error": "No groups found in sensitive feature (maybe all missing?)."}

        group_accuracy = {k: v["accuracy"] for k, v in groups.items()}
        group_recall = {k: v["recall"] for k, v in groups.items()}
        group_sel = {k: v["selection_rate"] for k, v in groups.items()}

        sel_vals = list(group_sel.values())
        dp_diff = float(max(sel_vals) - min(sel_vals)) if len(sel_vals) >= 2 else 0.0
        rec_vals = list(group_recall.values())
        eo_diff = float(max(rec_vals) - min(rec_vals)) if len(rec_vals) >= 2 else 0.0

        return {
            "Demographic Parity Difference": dp_diff,
            "Equal Opportunity Difference": eo_diff,
            "Group Accuracy": group_accuracy,
            "Group Recall": group_recall,
            "Group Selection Rate": group_sel,
            "classes": classes
        }
    except Exception as e:
        return {"error": f"Exception in compute_model_fairness: {str(e)}"}
