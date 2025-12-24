# modules/bias_utils.py
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from typing import Optional

def calc_mutual_info(df: pd.DataFrame, col: str, target_col: Optional[str]) -> Optional[float]:
    if target_col is None:
        return None
    try:
        a = df[col].astype(str).fillna("NAN")
        b = df[target_col].astype(str).fillna("NAN")
        return float(mutual_info_score(a, b))
    except Exception:
        return None

def calc_spd(df: pd.DataFrame, sensitive_col: str, target_col: Optional[str]) -> Optional[float]:
    if target_col is None:
        return None
    try:
        dfc = df.copy()
        if dfc[target_col].dtype == object:
            le = LabelEncoder()
            dfc[target_col] = le.fit_transform(dfc[target_col].astype(str))
        vals = dfc[sensitive_col].dropna().unique()
        if len(vals) < 2:
            return None
        spds = []
        for v in vals:
            mask = dfc[sensitive_col] == v
            p_priv = dfc[mask][target_col].mean()
            p_unpriv = dfc[~mask][target_col].mean()
            spds.append(float(p_unpriv - p_priv))
        return float(max(spds, key=abs))
    except Exception:
        return None

def calc_di(df: pd.DataFrame, sensitive_col: str, target_col: Optional[str]) -> Optional[float]:
    if target_col is None:
        return None
    try:
        dfc = df.copy()
        if dfc[target_col].dtype == object:
            le = LabelEncoder()
            dfc[target_col] = le.fit_transform(dfc[target_col].astype(str))
        vals = dfc[sensitive_col].dropna().unique()
        if len(vals) < 2:
            return None
        ratios = []
        for v in vals:
            mask = dfc[sensitive_col] == v
            p_priv = dfc[mask][target_col].mean()
            p_unpriv = dfc[~mask][target_col].mean()
            if p_priv == 0:
                continue
            ratios.append(p_unpriv / p_priv)
        if len(ratios) == 0:
            return None
        return float(min(ratios, key=lambda x: abs(x - 1)))
    except Exception:
        return None
