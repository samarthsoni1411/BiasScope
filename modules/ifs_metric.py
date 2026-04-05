# modules/ifs_metric.py
"""
Intersectional Fairness Score (IFS)
====================================
A novel, population-weighted composite metric for measuring intersectional fairness.

Unlike existing metrics (SPD, DI, Equal Opportunity Difference) that:
  (a) Report per-group disparities without aggregation, or
  (b) Use max-min differences that ignore subgroup size

IFS weights each subgroup's bias by its population fraction, ensuring that
bias affecting more people contributes proportionally more to the score.

Formal Definition
-----------------
Given dataset D = {(X_i, A_i, Y_i)} with n samples, where A produces m
intersectional subgroups S_1, ..., S_m with sizes n_1, ..., n_m:

  IFS(D, A, Y) = 1 - sum_{j=1}^{m} [ (n_j / n) * |DeltaP_j| ]

where DeltaP_j = P(Y=1 | S_j) - P(Y=1 | ~S_j)
            is the absolute Statistical Parity Difference for subgroup S_j.

Properties
----------
- Bounded: IFS ∈ [0, 1]
- IFS = 1 iff all subgroups share identical positive outcome rates (perfect fairness)
- IFS = 0 when population-weighted bias sums to 1 (maximum bias)
- Population-sensitive: bias affecting large subgroups penalizes IFS more
- Supports intersectional subgroups of depth k (not just single attributes)

Model-Level IFS
---------------
Replace Y (ground truth) with Y_hat (model predictions) to measure
model-induced intersectional fairness. Pre-mitigation vs post-mitigation
IFS difference quantifies the mitigation gain.

Authors: BiasScope Research Team
Year: 2026
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


# ─────────────────────────────────────────────────────────────────────────────
# Core IFS computation
# ─────────────────────────────────────────────────────────────────────────────

def _encode_binary_target(y: pd.Series) -> np.ndarray:
    """Encodes any binary target to {0, 1}."""
    y = pd.Series(y).reset_index(drop=True)
    if set(y.dropna().unique()).issubset({0, 1}):
        return y.fillna(0).astype(int).values
    le = LabelEncoder()
    encoded = le.fit_transform(y.astype(str))
    if len(le.classes_) == 2:
        return encoded
    # Multi-class: binarize vs mode
    mode_val = str(y.mode().iloc[0])
    return (y.astype(str) == mode_val).astype(int).values


def _create_intersectional_col(df: pd.DataFrame, sensitive_cols: List[str]) -> pd.Series:
    """Concatenates multiple sensitive columns into a single intersectional label."""
    return df[sensitive_cols].astype(str).agg("_".join, axis=1)


def compute_ifs(
    df: pd.DataFrame,
    sensitive_cols: List[str],
    target_col: str,
    min_group_size: int = 10,
    use_predictions: bool = False,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute the Intersectional Fairness Score (IFS).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing sensitive and target columns.
    sensitive_cols : List[str]
        One or more sensitive attribute column names.
        Multiple columns are combined into intersectional subgroups.
    target_col : str
        Name of the binary target column (ground truth).
    min_group_size : int
        Subgroups smaller than this are excluded from computation.
    use_predictions : bool
        If True, uses y_pred instead of ground truth (model-level IFS).
    y_pred : np.ndarray, optional
        Model predictions (required if use_predictions=True).

    Returns
    -------
    dict with keys:
        ifs           : float, the IFS score in [0, 1]
        subgroup_bias : dict mapping subgroup label → (SPD, population_weight)
        n_subgroups   : int, number of valid subgroups used
        excluded      : list of subgroups excluded for small sample size
        weighted_bias : float, the raw weighted bias sum (1 - IFS)
    """
    df = df.copy().reset_index(drop=True)

    # Build intersectional column
    if len(sensitive_cols) > 1:
        intersect_col = "__ifs_group__"
        df[intersect_col] = _create_intersectional_col(df, sensitive_cols)
    else:
        intersect_col = sensitive_cols[0]

    # Encode target
    y = _encode_binary_target(df[target_col])
    if use_predictions:
        if y_pred is None:
            raise ValueError("y_pred must be provided when use_predictions=True")
        y = _encode_binary_target(pd.Series(y_pred))

    n_total = len(y)
    groups = df[intersect_col].astype(str)

    subgroup_bias = {}
    excluded = []
    valid_groups = []

    group_counts = groups.value_counts()
    for grp, count in group_counts.items():
        if count < min_group_size:
            excluded.append(grp)
            continue
        valid_groups.append(grp)

    if len(valid_groups) < 2:
        return {
            "ifs": None,
            "subgroup_bias": {},
            "n_subgroups": 0,
            "excluded": excluded,
            "weighted_bias": None,
            "error": f"Fewer than 2 valid subgroups (min_group_size={min_group_size})"
        }

    # Compute per-group SPD and population weight
    weighted_bias_sum = 0.0
    for grp in valid_groups:
        mask = (groups == grp).values
        n_grp = mask.sum()
        w_j = n_grp / n_total  # population weight

        p_grp = y[mask].mean()              # P(Y=1 | S_j)
        p_others = y[~mask].mean()          # P(Y=1 | ~S_j)
        spd_j = float(abs(p_grp - p_others))

        weighted_bias_j = w_j * spd_j
        weighted_bias_sum += weighted_bias_j

        subgroup_bias[grp] = {
            "spd": spd_j,
            "population_weight": float(w_j),
            "positive_rate": float(p_grp),
            "weighted_contribution": float(weighted_bias_j),
            "n": int(n_grp),
        }

    # IFS = 1 - weighted bias sum, clamped to [0, 1]
    ifs_score = float(np.clip(1.0 - weighted_bias_sum, 0.0, 1.0))

    return {
        "ifs": ifs_score,
        "subgroup_bias": subgroup_bias,
        "n_subgroups": len(valid_groups),
        "excluded": excluded,
        "weighted_bias": float(weighted_bias_sum),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ifs_ci(
    df: pd.DataFrame,
    sensitive_cols: List[str],
    target_col: str,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    min_group_size: int = 10,
    use_predictions: bool = False,
    y_pred: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence intervals for IFS.

    Uses the percentile bootstrap method (Efron 1979).

    Returns
    -------
    dict with:
        ifs_mean   : float, mean IFS across bootstrap samples
        ci_lower   : float, lower bound of (ci*100)% confidence interval
        ci_upper   : float, upper bound of (ci*100)% confidence interval
        ci_width   : float, width of the confidence interval
        std        : float, standard deviation of bootstrap IFS values
    """
    rng = np.random.RandomState(random_state)
    boot_scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(df), size=len(df), replace=True)
        df_boot = df.iloc[idx].reset_index(drop=True)
        y_pred_boot = y_pred[idx] if y_pred is not None else None

        result = compute_ifs(
            df_boot, sensitive_cols, target_col,
            min_group_size=min_group_size,
            use_predictions=use_predictions,
            y_pred=y_pred_boot
        )
        if result.get("ifs") is not None:
            boot_scores.append(result["ifs"])

    if len(boot_scores) < 10:
        return {"ifs_mean": None, "ci_lower": None, "ci_upper": None,
                "ci_width": None, "std": None}

    boot_arr = np.array(boot_scores)
    alpha = 1.0 - ci
    lower = float(np.percentile(boot_arr, 100 * alpha / 2))
    upper = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))

    return {
        "ifs_mean": float(np.mean(boot_arr)),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_width": float(upper - lower),
        "std": float(np.std(boot_arr)),
        "n_valid_bootstrap": len(boot_scores),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Significance Test
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_ifs_test(
    ifs_before_samples: List[float],
    ifs_after_samples: List[float],
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test to assess whether mitigation
    significantly improved IFS.

    H0: Median IFS before == Median IFS after
    H1: IFS after > IFS before (one-tailed)

    Returns p-value and effect size (Cohen's d approximation).
    """
    try:
        from scipy import stats
        stat, p_two_tailed = stats.wilcoxon(ifs_after_samples, ifs_before_samples)
        p_one_tailed = p_two_tailed / 2  # directional test

        # Cohen's d for paired samples
        diff = np.array(ifs_after_samples) - np.array(ifs_before_samples)
        d = float(np.mean(diff) / (np.std(diff) + 1e-9))

        return {
            "statistic": float(stat),
            "p_value_two_tailed": float(p_two_tailed),
            "p_value_one_tailed": float(p_one_tailed),
            "significant_at_0.05": p_one_tailed < 0.05,
            "cohens_d": d,
            "effect_size": "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small",
        }
    except ImportError:
        return {"error": "scipy not installed — run: pip install scipy"}


# ─────────────────────────────────────────────────────────────────────────────
# Comparison helper: IFS vs traditional metrics
# ─────────────────────────────────────────────────────────────────────────────

def compare_metrics(
    df: pd.DataFrame,
    sensitive_cols: List[str],
    target_col: str,
    min_group_size: int = 10,
) -> pd.DataFrame:
    """
    Computes IFS, SPD (traditional, marginal), and DI for comparison.
    Returns a DataFrame showing where IFS provides more information.
    """
    from .bias_utils import calc_spd, calc_di, calc_mutual_info, create_intersectional_feature

    results = []

    # Traditional marginal metrics (one column at a time)
    for col in sensitive_cols:
        spd = calc_spd(df, col, target_col, min_samples=min_group_size)
        di  = calc_di(df, col, target_col, min_samples=min_group_size)
        mi  = calc_mutual_info(df, col, target_col)
        results.append({
            "Feature": col,
            "Type": "Marginal",
            "SPD": round(spd, 4) if spd is not None else None,
            "DI": round(di, 4) if di is not None else None,
            "MI": round(mi, 4) if mi is not None else None,
            "IFS": None,
        })

    # Intersectional IFS
    ifs_result = compute_ifs(df, sensitive_cols, target_col, min_group_size)
    results.append({
        "Feature": " × ".join(sensitive_cols),
        "Type": "Intersectional (IFS)",
        "SPD": None,
        "DI": None,
        "MI": None,
        "IFS": round(ifs_result["ifs"], 4) if ifs_result.get("ifs") is not None else None,
    })

    return pd.DataFrame(results)
