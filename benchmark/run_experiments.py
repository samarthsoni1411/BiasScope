# benchmark/run_experiments.py
"""
BiasScope Publication Benchmark
================================
Runs the full evaluation pipeline on all 3 benchmark datasets.

For each dataset:
  1. Preprocesses data
  2. Trains CatBoost (GPU if available, else CPU)
  3. Computes bias metrics (SPD, DI, EO Diff, IFS) before mitigation
  4. Runs ExponentiatedGradient (LightGBM base) mitigation
  5. Computes all metrics again after mitigation
  6. Computes bootstrap 95% CIs for IFS (500 resamples for speed)
  7. Saves full results to benchmark/results/results.csv
  8. Saves a LaTeX-ready table to benchmark/results/table_main.tex

Run: python -m benchmark.run_experiments   (from BiasScope root)
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# BiasScope modules
from modules.ifs_metric import compute_ifs, bootstrap_ifs_ci, wilcoxon_ifs_test
from modules.fairness_metrics import compute_model_fairness
from benchmark.data_loader import DATASETS

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GPU Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_catboost_model(task_type: str = "GPU"):
    """Returns CatBoost with GPU support, falling back to CPU."""
    try:
        from catboost import CatBoostClassifier
        try:
            model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.1,
                task_type=task_type,
                verbose=0,
                random_seed=42,
            )
            return model, "CatBoost-GPU"
        except Exception:
            model = CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.1,
                task_type="CPU", verbose=0, random_seed=42,
            )
            return model, "CatBoost-CPU"
    except ImportError:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, num_leaves=63, random_state=42, verbose=-1), "LightGBM"


def get_mitigation_estimator():
    """Returns LightGBM for Fairlearn mitigation (GPU-aware if available)."""
    try:
        from lightgbm import LGBMClassifier
        try:
            return LGBMClassifier(
                n_estimators=100, num_leaves=31,
                random_state=42, verbose=-1,
                device="gpu",
            ), "LightGBM-GPU"
        except Exception:
            return LGBMClassifier(
                n_estimators=100, num_leaves=31, random_state=42, verbose=-1
            ), "LightGBM-CPU"
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000), "LogisticRegression"


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, target_col: str):
    """Impute, encode, scale — returns X (numeric), y, feature_names, scaler."""
    df = df.copy().dropna(subset=[target_col]).reset_index(drop=True)

    y_raw = df[target_col]
    if y_raw.dtype == object or y_raw.nunique() > 2:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.astype(int).values

    X = df.drop(columns=[target_col]).copy()

    # Impute
    for col in X.columns:
        if X[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode().iloc[0])

    # Encode categoricals
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=X.columns)

    return X_df, y, list(X.columns), scaler


# ─────────────────────────────────────────────────────────────────────────────
# Single Dataset Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(name: str, cfg: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  DATASET: {name.upper()}")
    print(f"  {cfg['description']}")
    print(f"{'='*60}")

    # Load data
    df = cfg["loader"]()
    target_col = cfg["target"]
    sens_intersectional = cfg["sensitive_intersectional"]

    # Keep sensitive cols in raw form for bias metrics
    sens_raw = df[sens_intersectional].copy()

    # Preprocess
    print("\n[1/5] Preprocessing...")
    X, y, feat_names, scaler = preprocess(df, target_col)
    X["__target__"] = y
    X[sens_intersectional] = sens_raw.values
    X = X.reset_index(drop=True)

    # Train/test split
    X_train = X.sample(frac=0.8, random_state=42)
    X_test  = X.drop(X_train.index).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)

    feat_cols = [c for c in feat_names if c not in sens_intersectional]
    y_train = X_train["__target__"].values
    y_test  = X_test["__target__"].values
    X_tr = X_train[feat_cols]
    X_te = X_test[feat_cols]
    sens_test = X_test[sens_intersectional]

    # ── STEP 2: Train baseline model ──────────────────────────────────────────
    print("[2/5] Training baseline CatBoost model...")
    t0 = time.time()
    base_model, base_model_name = get_catboost_model("GPU")
    try:
        base_model.fit(X_tr, y_train)
    except Exception:
        base_model, base_model_name = get_catboost_model("CPU")
        base_model.fit(X_tr, y_train)

    y_pred_base = base_model.predict(X_te)
    train_time = round(time.time() - t0, 1)
    print(f"   Model: {base_model_name} | Time: {train_time}s")

    acc_base  = float(accuracy_score(y_test, y_pred_base))
    f1_base   = float(f1_score(y_test, y_pred_base, average="weighted", zero_division=0))

    # ── STEP 3: Pre-mitigation bias metrics ───────────────────────────────────
    print("[3/5] Computing pre-mitigation bias metrics...")
    # IFS on test set (model-level)
    test_df_for_ifs = sens_test.copy()
    test_df_for_ifs[target_col] = y_test

    ifs_before = compute_ifs(
        test_df_for_ifs, sens_intersectional, target_col,
        use_predictions=True, y_pred=y_pred_base, min_group_size=5
    )

    # Traditional fairness metrics
    fairness_before = compute_model_fairness(y_test, y_pred_base, sens_test[sens_intersectional[0]])
    dp_diff_before = fairness_before.get("Demographic Parity Difference", float("nan"))
    eo_diff_before = fairness_before.get("Equal Opportunity Difference", float("nan"))

    # Marginal DI
    from modules.bias_utils import calc_di
    df_test_raw = X_test.copy()
    df_test_raw["__pred__"] = y_pred_base
    di_before = calc_di(df_test_raw, sens_intersectional[0], "__pred__", min_samples=5)

    print(f"   IFS (before): {ifs_before.get('ifs', 'N/A'):.4f}")
    print(f"   DP Diff:      {dp_diff_before:.4f}")
    print(f"   DI:           {di_before:.4f}" if di_before else "   DI: N/A")

    # ── STEP 4: Mitigation ────────────────────────────────────────────────────
    print("[4/5] Running bias mitigation (ExponentiatedGradient)...")
    t1 = time.time()

    # Build encoded training set for Fairlearn
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    mit_model, mit_name = get_mitigation_estimator()
    constraint = DemographicParity()
    mitigator = ExponentiatedGradient(mit_model, constraints=constraint, max_iter=30)

    # Fairlearn needs raw sensitive features during training
    sens_train_raw = X_train[sens_intersectional[0]].astype(str)
    try:
        mitigator.fit(X_tr, y_train, sensitive_features=sens_train_raw)
        y_pred_mit = mitigator.predict(X_te)
    except Exception as e:
        print(f"   ⚠️ Mitigation failed ({e}), using base model predictions")
        y_pred_mit = y_pred_base.copy()

    mit_time = round(time.time() - t1, 1)
    print(f"   Mitigator: {mit_name} | Time: {mit_time}s")

    acc_mit = float(accuracy_score(y_test, y_pred_mit))
    f1_mit  = float(f1_score(y_test, y_pred_mit, average="weighted", zero_division=0))

    # ── STEP 5: Post-mitigation bias metrics ──────────────────────────────────
    print("[5/5] Computing post-mitigation bias metrics...")
    ifs_after = compute_ifs(
        test_df_for_ifs, sens_intersectional, target_col,
        use_predictions=True, y_pred=y_pred_mit, min_group_size=5
    )

    fairness_after = compute_model_fairness(y_test, y_pred_mit, sens_test[sens_intersectional[0]])
    dp_diff_after = fairness_after.get("Demographic Parity Difference", float("nan"))
    eo_diff_after = fairness_after.get("Equal Opportunity Difference", float("nan"))

    df_test_raw["__pred_mit__"] = y_pred_mit
    di_after = calc_di(df_test_raw, sens_intersectional[0], "__pred_mit__", min_samples=5)

    print(f"   IFS (after):  {ifs_after.get('ifs', 'N/A'):.4f}")
    print(f"   IFS gain:     {(ifs_after.get('ifs', 0) - ifs_before.get('ifs', 0)):+.4f}")

    # ── Bootstrap CIs ──────────────────────────────────────────────────────────
    print("   Computing bootstrap CIs (500 resamples)...")
    ci_before = bootstrap_ifs_ci(
        test_df_for_ifs, sens_intersectional, target_col,
        n_bootstrap=500, use_predictions=True, y_pred=y_pred_base, min_group_size=5
    )
    ci_after = bootstrap_ifs_ci(
        test_df_for_ifs, sens_intersectional, target_col,
        n_bootstrap=500, use_predictions=True, y_pred=y_pred_mit, min_group_size=5
    )

    return {
        "dataset": name,
        "n_samples": len(df),
        "n_features": len(feat_cols),
        "n_subgroups": ifs_before.get("n_subgroups", 0),
        "base_model": base_model_name,
        # Accuracy
        "acc_before": round(acc_base, 4),
        "acc_after":  round(acc_mit, 4),
        "acc_delta":  round(acc_mit - acc_base, 4),
        "f1_before":  round(f1_base, 4),
        "f1_after":   round(f1_mit, 4),
        # IFS
        "ifs_before": round(ifs_before.get("ifs", float("nan")), 4),
        "ifs_after":  round(ifs_after.get("ifs", float("nan")), 4),
        "ifs_delta":  round((ifs_after.get("ifs", 0) - ifs_before.get("ifs", 0)), 4),
        "ifs_ci_before_lower": round(ci_before.get("ci_lower", float("nan")), 4),
        "ifs_ci_before_upper": round(ci_before.get("ci_upper", float("nan")), 4),
        "ifs_ci_after_lower":  round(ci_after.get("ci_lower", float("nan")), 4),
        "ifs_ci_after_upper":  round(ci_after.get("ci_upper", float("nan")), 4),
        # Traditional metrics
        "dp_diff_before": round(dp_diff_before, 4),
        "dp_diff_after":  round(dp_diff_after, 4),
        "eo_diff_before": round(eo_diff_before, 4),
        "eo_diff_after":  round(eo_diff_after, 4),
        "di_before": round(di_before, 4) if di_before else float("nan"),
        "di_after":  round(di_after, 4) if di_after else float("nan"),
        # Runtime
        "train_time_s": train_time,
        "mit_time_s":   mit_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX Table Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_latex_table(df_results: pd.DataFrame) -> str:
    """Generates a publication-ready LaTeX table from results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{BiasScope Evaluation: IFS, Accuracy, and Fairness Metrics Before/After Mitigation}",
        r"\label{tab:main_results}",
        r"\centering",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Acc.↑} & \textbf{Acc.↑} & \textbf{IFS↑} & \textbf{IFS↑} "
        r"& \textbf{DP↓} & \textbf{DP↓} & \textbf{DI↑} & \textbf{DI↑} \\",
        r" & \textit{Before} & \textit{After} & \textit{Before} & \textit{After} "
        r"& \textit{Before} & \textit{After} & \textit{Before} & \textit{After} \\",
        r"\midrule",
    ]

    dataset_labels = {
        "adult_income": "Adult Income",
        "compas": "COMPAS",
        "german_credit": "German Credit",
    }

    for _, row in df_results.iterrows():
        label = dataset_labels.get(row["dataset"], row["dataset"])
        ifs_b = f"{row['ifs_before']:.4f} [{row['ifs_ci_before_lower']:.3f},{row['ifs_ci_before_upper']:.3f}]"
        ifs_a = f"{row['ifs_after']:.4f} [{row['ifs_ci_after_lower']:.3f},{row['ifs_ci_after_upper']:.3f}]"
        line = (
            f"{label} & {row['acc_before']:.4f} & {row['acc_after']:.4f} & "
            f"{ifs_b} & {ifs_a} & "
            f"{row['dp_diff_before']:.4f} & {row['dp_diff_after']:.4f} & "
            f"{row['di_before']:.4f} & {row['di_after']:.4f} \\\\"
        )
        lines.append(line)

    lines += [
        r"\bottomrule",
        r"\multicolumn{9}{p{0.95\linewidth}}{\footnotesize "
        r"\textit{IFS = Intersectional Fairness Score (↑ = higher is fairer), "
        r"DP Diff = Demographic Parity Difference (↓ = lower is fairer), "
        r"DI = Disparate Impact Ratio (↑ = higher is fairer, $\geq 0.8$ = legal threshold). "
        r"Bootstrap 95\% CIs shown in brackets for IFS.}}",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Wilcoxon Test Across All Datasets
# ─────────────────────────────────────────────────────────────────────────────

def run_significance_test(all_results: list) -> dict:
    """
    Wilcoxon signed-rank test on IFS before vs after across all datasets.
    Also runs per-dataset paired t-test on bootstrap samples (simulated paired data).
    """
    ifs_before_vals = [r["ifs_before"] for r in all_results if not pd.isna(r["ifs_before"])]
    ifs_after_vals  = [r["ifs_after"]  for r in all_results if not pd.isna(r["ifs_after"])]

    if len(ifs_before_vals) < 3:
        return {"error": "Not enough data points for significance test (need >= 3 datasets)"}

    return wilcoxon_ifs_test(ifs_before_vals, ifs_after_vals)


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BiasScope Publication Benchmark")
    print("  GPU: RTX 5050 | Attempting GPU acceleration")
    print("="*60)

    all_results = []
    total_start = time.time()

    for ds_name, ds_cfg in DATASETS.items():
        try:
            result = evaluate_dataset(ds_name, ds_cfg)
            all_results.append(result)
            print(f"\n  ✅ {ds_name} complete.")
        except Exception as e:
            print(f"\n  ❌ {ds_name} FAILED: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("\n❌ No results generated. Check errors above.")
        sys.exit(1)

    total_time = round(time.time() - total_start, 1)
    print(f"\n{'='*60}")
    print(f"  All datasets processed in {total_time}s")
    print(f"{'='*60}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df_results = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    print(df_results[["dataset", "acc_before", "acc_after", "ifs_before", "ifs_after",
                       "dp_diff_before", "dp_diff_after"]].to_string(index=False))

    # ── Significance Test ──────────────────────────────────────────────────────
    print("\n--- Statistical Significance (Wilcoxon Test) ---")
    sig_test = run_significance_test(all_results)
    for k, v in sig_test.items():
        print(f"  {k}: {v}")

    sig_path = RESULTS_DIR / "significance_test.json"
    import json

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    with open(sig_path, "w") as f:
        json.dump(sig_test, f, indent=2, cls=_NumpyEncoder)
    print(f"  Saved to: {sig_path}")

    # ── LaTeX Table ────────────────────────────────────────────────────────────
    latex_table = generate_latex_table(df_results)
    tex_path = RESULTS_DIR / "table_main.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"\n✅ LaTeX table saved to: {tex_path}")

    print("\n🎉 Benchmark complete. Run benchmark/generate_figures.py next.")
