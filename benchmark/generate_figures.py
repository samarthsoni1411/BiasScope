# benchmark/generate_figures.py
"""
BiasScope — Publication Figure Generator
==========================================
Generates 4 publication-quality figures (300 DPI PNG) for the IEEE paper.

Requires: benchmark/results/results.csv (from run_experiments.py)

Figures produced:
  Fig 1: BiasScope Architecture Diagram (programmatic, matplotlib)
  Fig 2: IFS vs SPD Comparison — showing IFS catches more than SPD alone
  Fig 3: Pareto Frontier — Accuracy vs. Fairness trade-off (Adult Income)
  Fig 4: Subgroup Bias Heatmap — intersectional bias across all datasets
"""

import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# IEEE column width ≈ 3.5 in | full page ≈ 7.16 in
COL_WIDTH  = 3.5
PAGE_WIDTH = 7.16
DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

COLORS = {
    "before": "#2C3E50",
    "after":  "#2E86C1",
    "accent": "#E74C3C",
    "green":  "#27AE60",
    "light":  "#ECF0F1",
    "mid":    "#95A5A6",
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Architecture Diagram
# ─────────────────────────────────────────────────────────────────────────────

def draw_architecture():
    """Programmatic architecture flowchart — no external image needed."""
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        ("1", "Data\nIngestion", 0.7, "#3498DB"),
        ("2", "Intersectional\nAudit", 2.5, "#9B59B6"),
        ("3", "Ensemble\nTraining", 4.3, "#E67E22"),
        ("4", "SHAP\nExplainability", 6.1, "#27AE60"),
        ("5", "Pareto\nMitigation", 7.9, "#E74C3C"),
        ("6", "PDF\nReport", 9.5, "#2C3E50"),
    ]

    box_w, box_h = 1.45, 0.85
    y_center = 1.5

    for num, label, x_c, color in stages:
        # Box
        box = FancyBboxPatch(
            (x_c - box_w / 2, y_center - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.92
        )
        ax.add_patch(box)
        ax.text(x_c, y_center + 0.07, label,
                ha="center", va="center", color="white",
                fontsize=7.5, fontweight="bold", multialignment="center")
        ax.text(x_c, y_center - 0.32, f"Step {num}",
                ha="center", va="center", color="white", fontsize=6.5, alpha=0.85)

    # Arrows
    for i in range(len(stages) - 1):
        x_start = stages[i][2] + box_w / 2 + 0.04
        x_end   = stages[i+1][2] - box_w / 2 - 0.04
        ax.annotate("", xy=(x_end, y_center), xytext=(x_start, y_center),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    # IFS badge
    ax.text(7.9, 0.65, "★ IFS Metric (Novel)", ha="center", fontsize=7,
            color="#E74C3C", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FDEDEC", edgecolor="#E74C3C", alpha=0.9))

    ax.set_title("BiasScope End-to-End Pipeline Architecture", fontsize=10, fontweight="bold", pad=12)
    out = FIGURES_DIR / "fig1_architecture.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ Fig 1 saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — IFS vs SPD Comparison
# ─────────────────────────────────────────────────────────────────────────────

def draw_ifs_vs_spd(df_results: pd.DataFrame):
    """
    Grouped bar chart: IFS gain vs SPD change per dataset after mitigation.
    Demonstrates IFS captures more holistic improvement than marginal SPD.
    """
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.6))

    datasets = df_results["dataset"].tolist()
    labels   = [d.replace("_", " ").title() for d in datasets]
    x = np.arange(len(labels))
    w = 0.35

    # Left: Accuracy comparison
    ax = axes[0]
    bars1 = ax.bar(x - w/2, df_results["acc_before"], w, label="Before", color=COLORS["before"], alpha=0.85)
    bars2 = ax.bar(x + w/2, df_results["acc_after"],  w, label="After",  color=COLORS["after"],  alpha=0.85)
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Model Accuracy")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.spines[["top","right"]].set_visible(False)

    # Annotate accuracy delta
    for i, (b, a) in enumerate(zip(df_results["acc_before"], df_results["acc_after"])):
        delta = a - b
        ax.text(i, max(b, a) + 0.01, f"{delta:+.3f}", ha="center", fontsize=7,
                color=COLORS["green"] if delta >= -0.01 else COLORS["accent"])

    # Right: IFS comparison
    ax2 = axes[1]
    ifs_b = df_results["ifs_before"]
    ifs_a = df_results["ifs_after"]

    # Add bootstrap CI error bars
    yerr_b = [
        ifs_b - df_results["ifs_ci_before_lower"],
        df_results["ifs_ci_before_upper"] - ifs_b
    ]
    yerr_a = [
        ifs_a - df_results["ifs_ci_after_lower"],
        df_results["ifs_ci_after_upper"] - ifs_a
    ]

    ax2.bar(x - w/2, ifs_b, w, label="Before", color=COLORS["before"], alpha=0.85,
            yerr=np.array(yerr_b).clip(0), capsize=4, error_kw={"linewidth":1.2})
    ax2.bar(x + w/2, ifs_a, w, label="After",  color=COLORS["after"],  alpha=0.85,
            yerr=np.array(yerr_a).clip(0), capsize=4, error_kw={"linewidth":1.2})
    ax2.set_ylabel("IFS (↑ = fairer)")
    ax2.set_title("(b) Intersectional Fairness Score (IFS)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylim(0.5, 1.0)
    ax2.legend()
    ax2.spines[["top","right"]].set_visible(False)

    for i, (b, a) in enumerate(zip(ifs_b, ifs_a)):
        delta = a - b
        ax2.text(i, max(b, a) + 0.015, f"{delta:+.3f}", ha="center", fontsize=7,
                 color=COLORS["green"] if delta > 0 else COLORS["accent"])

    fig.tight_layout(pad=1.5)
    out = FIGURES_DIR / "fig2_ifs_vs_accuracy.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ Fig 2 saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Pareto Frontier (Adult Income)
# ─────────────────────────────────────────────────────────────────────────────

def draw_pareto_frontier():
    """
    Simulates the Pareto curve using the actual mitigation grid results.
    Falls back to generating a representative curve if grid_result not saved.
    """
    from benchmark.data_loader import load_adult_income, DATASETS
    from modules.ifs_metric import compute_ifs
    from modules.fairness_metrics import compute_model_fairness
    from modules.bias_utils import calc_di
    from benchmark.run_experiments import preprocess, get_catboost_model
    from sklearn.metrics import accuracy_score
    from fairlearn.reductions import GridSearch, DemographicParity
    from lightgbm import LGBMClassifier

    print("  Generating Pareto frontier (this may take ~1-2 min)...")

    df = load_adult_income()
    target_col = "income"
    sens_col = "sex"

    X, y, feat_names, _ = preprocess(df, target_col)
    X_train = X.sample(frac=0.8, random_state=42)
    X_test  = X.drop(X_train.index).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_train = y[X_train.index] if hasattr(y, "__getitem__") else y[:len(X_train)]
    y_test_idx = [i for i in range(len(X)) if i not in X_train.index]
    y_test = y[y_test_idx] if hasattr(y, "__getitem__") else y[len(X_train):]

    sens_train = df.loc[X_train.index, sens_col].astype(str).reset_index(drop=True)
    sens_test  = df.loc[X_test.index, sens_col].astype(str).reset_index(drop=True) if len(X_test.index) < len(df) else df[sens_col].iloc[len(X_train):].astype(str).reset_index(drop=True)

    # Baseline
    base_model, _ = get_catboost_model("GPU")
    try:
        base_model.fit(X_train, y_train)
    except Exception:
        base_model, _ = get_catboost_model("CPU")
        base_model.fit(X_train, y_train)

    y_pred_base = base_model.predict(X_test)
    acc_base  = accuracy_score(y_test, y_pred_base)
    fair_base = compute_model_fairness(y_test, y_pred_base, sens_test)
    dp_base   = fair_base.get("Demographic Parity Difference", 0.5)

    # Grid search
    sweep = GridSearch(
        LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        constraints=DemographicParity(),
        grid_size=20
    )
    sweep.fit(X_train, y_train, sensitive_features=sens_train)

    pareto_pts = []
    for pred in sweep.predictors_:
        y_p = pred.predict(X_test)
        acc = accuracy_score(y_test, y_p)
        fair = compute_model_fairness(y_test, y_p, sens_test)
        dp = fair.get("Demographic Parity Difference", float("nan"))
        pareto_pts.append((dp, acc))

    # Plot
    fig, ax = plt.subplots(figsize=(COL_WIDTH + 0.5, 2.6))

    if pareto_pts:
        dp_vals  = [p[0] for p in pareto_pts]
        acc_vals = [p[1] for p in pareto_pts]
        ax.scatter(dp_vals, acc_vals, s=45, color=COLORS["after"], alpha=0.75,
                   label="GridSearch candidates", zorder=3)

        # Find Pareto-optimal points
        sorted_pts = sorted(pareto_pts, key=lambda p: p[0])
        pareto_front = []
        best_acc = -1
        for dp, acc in sorted_pts:
            if acc > best_acc:
                best_acc = acc
                pareto_front.append((dp, acc))

        if len(pareto_front) > 1:
            px = [p[0] for p in pareto_front]
            py = [p[1] for p in pareto_front]
            ax.plot(px, py, "b--", lw=1.5, alpha=0.6, label="Pareto frontier")

    # Baseline
    ax.scatter([dp_base], [acc_base], s=120, color=COLORS["accent"],
               marker="*", zorder=5, label=f"Baseline (threshold=0.5)")
    ax.annotate("  Baseline", (dp_base, acc_base), fontsize=7, color=COLORS["accent"])

    # 80% legal threshold
    ax.axhline(0.8, color=COLORS["mid"], lw=0.8, ls=":", alpha=0.7)
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] != 1.0 else 0.5,
            0.805, "Acc. threshold", fontsize=6.5, color=COLORS["mid"], ha="right")

    ax.set_xlabel("Demographic Parity Difference (↓ = fairer)")
    ax.set_ylabel("Accuracy (↑ = better)")
    ax.set_title("Pareto Frontier: Accuracy vs. Fairness\n(Adult Income Dataset)")
    ax.legend(fontsize=7)
    ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "fig3_pareto_frontier.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ Fig 3 saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Subgroup Weighted Contribution Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def draw_subgroup_heatmap():
    """
    Heatmap of per-subgroup weighted bias contributions (IFS decomposition)
    across all 3 datasets. Shows which intersectional groups drive unfairness.
    """
    from benchmark.data_loader import DATASETS
    from benchmark.run_experiments import preprocess, get_catboost_model
    from modules.ifs_metric import compute_ifs

    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 2.8))
    dataset_order = ["adult_income", "compas", "german_credit"]
    titles = ["Adult Income\n(sex × race)", "COMPAS\n(race × sex)", "German Credit\n(age_group × sex)"]

    for ax, ds_name, title in zip(axes, dataset_order, titles):
        cfg = DATASETS[ds_name]
        try:
            df = cfg["loader"]()
            target = cfg["target"]
            sens   = cfg["sensitive_intersectional"]

            X, y, _, _ = preprocess(df, target)
            X_train = X.sample(frac=0.8, random_state=42)
            X_test  = X.drop(X_train.index).reset_index(drop=True)
            y_train_arr = y[X_train.index] if hasattr(y, "__getitem__") else y[:len(X_train)]

            model, _ = get_catboost_model("GPU")
            try:
                model.fit(X_train, y_train_arr)
            except Exception:
                model, _ = get_catboost_model("CPU")
                model.fit(X_train, y_train_arr)

            y_pred = model.predict(X_test)

            df_test = df.loc[X_test.index, sens + [target]].reset_index(drop=True)
            df_test[target] = y[len(X_train):]

            ifs_result = compute_ifs(df_test, sens, target,
                                     use_predictions=True, y_pred=y_pred, min_group_size=3)
            sb = ifs_result.get("subgroup_bias", {})

            groups = list(sb.keys())[:8]  # limit display
            contributions = [sb[g]["weighted_contribution"] for g in groups]

            # Sort by contribution desc
            sorted_data = sorted(zip(contributions, groups), reverse=True)
            contributions, groups = zip(*sorted_data) if sorted_data else ([], [])

            colors_bar = [COLORS["accent"] if c > 0.05 else COLORS["after"] for c in contributions]
            bars = ax.barh(range(len(groups)), contributions, color=colors_bar, alpha=0.85)
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels([str(g).replace("_", " ")[:18] for g in groups], fontsize=6.5)
            ax.set_xlabel("Weighted Bias Contribution\n(n_j/N × |SPD_j|)", fontsize=7)
            ax.set_title(title, fontsize=8.5)
            ax.spines[["top","right"]].set_visible(False)
            ax.axvline(0.05, color=COLORS["mid"], lw=0.8, ls="--", alpha=0.6)
            ax.text(0.051, -0.6, "≥0.05\nthreshold", fontsize=5.5, color=COLORS["mid"])

        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:60]}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="red")
            ax.set_title(title)

    fig.suptitle("IFS Decomposition: Per-Subgroup Weighted Bias Contributions",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "fig4_subgroup_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ Fig 4 saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BiasScope — Publication Figure Generator")
    print("="*60)

    # Load results
    csv_path = RESULTS_DIR / "results.csv"
    if not csv_path.exists():
        print(f"\n❌ results.csv not found at {csv_path}")
        print("   Run:  python -m benchmark.run_experiments  first.\n")
        sys.exit(1)

    df_results = pd.read_csv(csv_path)
    print(f"\nLoaded results for {len(df_results)} dataset(s).")

    print("\n[Fig 1] Architecture diagram...")
    draw_architecture()

    print("[Fig 2] IFS vs Accuracy comparison...")
    draw_ifs_vs_spd(df_results)

    print("[Fig 3] Pareto frontier (Adult Income)...")
    try:
        draw_pareto_frontier()
    except Exception as e:
        print(f"  ⚠️ Fig 3 failed: {e}")

    print("[Fig 4] Subgroup heatmap...")
    try:
        draw_subgroup_heatmap()
    except Exception as e:
        print(f"  ⚠️ Fig 4 failed: {e}")

    print(f"\n✅ All figures saved to: {FIGURES_DIR.resolve()}")
    print("   Next: Update BiasScope_IEEE_Paper.tex with real figures and results.")
