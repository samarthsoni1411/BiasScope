# modules/bias_utils.py
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from typing import Optional, List

def create_intersectional_feature(df: pd.DataFrame, sensitive_cols: List[str]) -> pd.Series:
    """
    Combines multiple sensitive columns into a single intersectional feature.
    Example: ['Gender', 'Race'] -> 'Male_White', 'Female_Black', etc.
    """
    return df[sensitive_cols].astype(str).agg('_'.join, axis=1)

def calc_mutual_info(df: pd.DataFrame, col: str, target_col: Optional[str]) -> Optional[float]:
    if target_col is None:
        return None
    try:
        a = df[col].astype(str).fillna("NAN")
        b = df[target_col].astype(str).fillna("NAN")
        return float(mutual_info_score(a, b))
    except Exception:
        return None

def calc_spd(df: pd.DataFrame, sensitive_col: str, target_col: Optional[str], min_samples: int = 10) -> Optional[float]:
    """
    Updated SPD with a minimum sample threshold for intersectional groups.
    """
    if target_col is None:
        return None
    try:
        dfc = df.copy()
        if dfc[target_col].dtype == object:
            le = LabelEncoder()
            dfc[target_col] = le.fit_transform(dfc[target_col].astype(str))
        
        # Filter out groups that are too small to be statistically significant
        counts = dfc[sensitive_col].value_counts()
        valid_groups = counts[counts >= min_samples].index.tolist()
        
        if len(valid_groups) < 2:
            return None # Not enough data for comparison
            
        df_filtered = dfc[dfc[sensitive_col].isin(valid_groups)]
        spds = []
        for v in valid_groups:
            mask = df_filtered[sensitive_col] == v
            p_v = df_filtered[mask][target_col].mean()
            p_others = df_filtered[~mask][target_col].mean()
            spds.append(float(p_others - p_v))
            
        return float(max(spds, key=abs))
    except Exception:
        return None

def calc_di(df: pd.DataFrame, sensitive_col: str, target_col: Optional[str], min_samples: int = 10) -> Optional[float]:
    """
    Updated Disparate Impact with a minimum sample threshold for intersectional groups.
    """
    if target_col is None:
        return None
    try:
        dfc = df.copy()
        if dfc[target_col].dtype == object:
            le = LabelEncoder()
            dfc[target_col] = le.fit_transform(dfc[target_col].astype(str))
            
        counts = dfc[sensitive_col].value_counts()
        valid_groups = counts[counts >= min_samples].index.tolist()
        
        if len(valid_groups) < 2:
            return None
            
        df_filtered = dfc[dfc[sensitive_col].isin(valid_groups)]
        ratios = []
        for v in valid_groups:
            mask = df_filtered[sensitive_col] == v
            p_v = df_filtered[mask][target_col].mean()
            p_others = df_filtered[~mask][target_col].mean()
            
            if p_others == 0:
                continue
            ratios.append(p_v / p_others)
            
        if len(ratios) == 0:
            return None
        return float(min(ratios, key=lambda x: abs(x - 1)))
    except Exception:
        return None