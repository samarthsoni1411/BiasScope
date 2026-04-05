# pages/dataset_comparison.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from modules.data_utils import save_uploaded_file, read_dataset, guess_target_column
from modules.bias_utils import (
    calc_spd, calc_di, calc_mutual_info, create_intersectional_feature
)

st.set_page_config(page_title="BiasScope | Dataset Comparison", layout="wide")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#2E86C1;'>🔀 Dataset Comparison Mode</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:gray; font-size:15px;'>"
    "Compare bias metrics, subgroup distributions, and fairness drift between two datasets "
    "(e.g. training data vs. live/production data, or pre- vs. post-collection)."
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 1 — LOAD DATASETS
# ─────────────────────────────────────────────
st.subheader("📂 Step 1 — Load Two Datasets")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Dataset A")
    use_session = st.checkbox(
        "Use currently uploaded dataset (from session)",
        value=bool(st.session_state.get("dataset_path"))
    )
    if use_session and st.session_state.get("dataset_path"):
        path_a = st.session_state["dataset_path"]
        st.success(f"✅ Using session dataset: `{path_a.split('/')[-1].split(chr(92))[-1]}`")
    else:
        file_a = st.file_uploader("Upload Dataset A (.csv / .xlsx)", type=["csv", "xlsx"], key="cmp_a")
        if file_a:
            path_a = save_uploaded_file(file_a)
            st.success(f"✅ Saved: `{file_a.name}`")
        else:
            path_a = None

with col_b:
    st.markdown("#### Dataset B")
    file_b = st.file_uploader("Upload Dataset B (.csv / .xlsx)", type=["csv", "xlsx"], key="cmp_b")
    if file_b:
        path_b = save_uploaded_file(file_b)
        st.success(f"✅ Saved: `{file_b.name}`")
    else:
        path_b = None

if not path_a or not path_b:
    st.info("👆 Please provide both datasets to begin the comparison.")
    st.stop()

# Load both dataframes
df_a = read_dataset(path_a)
df_b = read_dataset(path_b)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 2 — COLUMN CONFIGURATION
# ─────────────────────────────────────────────
st.subheader("⚙️ Step 2 — Configure Columns")

common_cols_a = list(df_a.columns)
common_cols_b = list(df_b.columns)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Dataset A settings**")
    guess_a = guess_target_column(df_a)
    target_a = st.selectbox(
        "Target column (A)",
        options=common_cols_a,
        index=common_cols_a.index(guess_a) if guess_a in common_cols_a else len(common_cols_a) - 1,
        key="ta"
    )
    sensitive_cols_a = st.multiselect(
        "Sensitive features (A)",
        options=[c for c in common_cols_a if c != target_a],
        key="sa"
    )

with col2:
    st.markdown("**Dataset B settings**")
    guess_b = guess_target_column(df_b)
    target_b = st.selectbox(
        "Target column (B)",
        options=common_cols_b,
        index=common_cols_b.index(guess_b) if guess_b in common_cols_b else len(common_cols_b) - 1,
        key="tb"
    )
    sensitive_cols_b = st.multiselect(
        "Sensitive features (B)",
        options=[c for c in common_cols_b if c != target_b],
        default=[c for c in sensitive_cols_a if c in common_cols_b],  # mirror A's selection
        key="sb"
    )

min_n = st.slider("Minimum group size (n_min)", 5, 100, 20)

if not sensitive_cols_a or not sensitive_cols_b:
    st.info("💡 Select at least one sensitive feature for each dataset.")
    st.stop()

# ─────────────────────────────────────────────
# HELPER — compute metrics for one dataset
# ─────────────────────────────────────────────
def compute_bias_metrics(df: pd.DataFrame, sensitive_cols: list, target_col: str, min_n: int):
    """Returns (audit_col_name, spd, di, mi, group_counts)"""
    if len(sensitive_cols) > 1:
        audit_col = "intersectional_feature"
        df = df.copy()
        df[audit_col] = create_intersectional_feature(df, sensitive_cols)
    else:
        audit_col = sensitive_cols[0]

    spd = calc_spd(df, audit_col, target_col, min_samples=min_n)
    di  = calc_di(df, audit_col, target_col, min_samples=min_n)
    mi  = calc_mutual_info(df, audit_col, target_col)
    counts = df[audit_col].value_counts()
    return audit_col, spd, di, mi, counts

# ─────────────────────────────────────────────
# SECTION 3 — RUN COMPARISON
# ─────────────────────────────────────────────
st.markdown("---")
if st.button("🚀 Run Bias Comparison", use_container_width=True, type="primary"):
    with st.spinner("Computing bias metrics for both datasets..."):
        try:
            col_a_name, spd_a, di_a, mi_a, counts_a = compute_bias_metrics(
                df_a, sensitive_cols_a, target_a, min_n
            )
            col_b_name, spd_b, di_b, mi_b, counts_b = compute_bias_metrics(
                df_b, sensitive_cols_b, target_b, min_n
            )
        except Exception as e:
            st.error(f"❌ Error computing metrics: {e}")
            st.stop()

    st.session_state["cmp_results"] = {
        "spd_a": spd_a, "di_a": di_a, "mi_a": mi_a, "counts_a": counts_a,
        "spd_b": spd_b, "di_b": di_b, "mi_b": mi_b, "counts_b": counts_b,
        "col_a": col_a_name, "col_b": col_b_name,
        "label_a": "Dataset A", "label_b": "Dataset B",
        "df_a": df_a, "df_b": df_b,
        "target_a": target_a, "target_b": target_b,
        "sensitive_a": sensitive_cols_a, "sensitive_b": sensitive_cols_b,
    }
    st.success("✅ Comparison complete! See results below.")

# ─────────────────────────────────────────────
# SECTION 4 — DISPLAY RESULTS
# ─────────────────────────────────────────────
if "cmp_results" not in st.session_state:
    st.stop()

R = st.session_state["cmp_results"]
spd_a, di_a, mi_a  = R["spd_a"], R["di_a"], R["mi_a"]
spd_b, di_b, mi_b  = R["spd_b"], R["di_b"], R["mi_b"]
counts_a, counts_b = R["counts_a"], R["counts_b"]

st.markdown("## 📊 Comparison Results")

# ── 4.1 Metric Cards ──────────────────────────────────────────────────────────
def _fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"

def _delta_str(a, b, lower_is_better=True):
    """Returns a human-readable drift string with arrow emoji."""
    if a is None or b is None:
        return "—"
    delta = b - a
    if abs(delta) < 0.005:
        return f"≈ No change ({delta:+.4f})"
    if lower_is_better:
        arrow = "🔴 Worse" if delta > 0 else "🟢 Better"
    else:
        arrow = "🟢 Better" if delta > 0 else "🔴 Worse"
    return f"{arrow} ({delta:+.4f})"

st.subheader("📌 Key Bias Metrics — Side by Side")
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("**Statistical Parity Difference (SPD)**")
    st.markdown(f"- Dataset A: `{_fmt(spd_a)}`")
    st.markdown(f"- Dataset B: `{_fmt(spd_b)}`")
    st.markdown(f"- Drift: {_delta_str(spd_a, spd_b, lower_is_better=True)}")
    st.caption("Lower SPD = more fair. Range: [0, 1]")

with m2:
    st.markdown("**Disparate Impact Ratio (DI)**")
    st.markdown(f"- Dataset A: `{_fmt(di_a)}`")
    st.markdown(f"- Dataset B: `{_fmt(di_b)}`")
    st.markdown(f"- Drift: {_delta_str(di_a, di_b, lower_is_better=False)}")
    st.caption("Higher DI = more fair. Legal threshold: ≥ 0.80")

with m3:
    st.markdown("**Mutual Information (MI)**")
    st.markdown(f"- Dataset A: `{_fmt(mi_a)}`")
    st.markdown(f"- Dataset B: `{_fmt(mi_b)}`")
    st.markdown(f"- Drift: {_delta_str(mi_a, mi_b, lower_is_better=True)}")
    st.caption("Lower MI = sensitive feature less predictive of outcome")

st.markdown("---")

# ── 4.2 Grouped Bar Chart — Metrics Comparison ───────────────────────────────
st.subheader("📈 Metrics Overview — Grouped Bar Chart")

metrics_vals = {
    "SPD (lower=fair)": [
        spd_a if spd_a is not None else 0,
        spd_b if spd_b is not None else 0
    ],
    "1 - DI (lower=fair)": [
        max(0, 1 - di_a) if di_a is not None else 0,
        max(0, 1 - di_b) if di_b is not None else 0
    ],
    "Mutual Info": [
        mi_a if mi_a is not None else 0,
        mi_b if mi_b is not None else 0
    ],
}

fig_bar = go.Figure()
colors = ["#2E86C1", "#E74C3C"]
for i, label in enumerate(["Dataset A", "Dataset B"]):
    fig_bar.add_trace(go.Bar(
        name=label,
        x=list(metrics_vals.keys()),
        y=[v[i] for v in metrics_vals.values()],
        marker_color=colors[i],
        text=[f"{v[i]:.4f}" for v in metrics_vals.values()],
        textposition="outside",
    ))

fig_bar.update_layout(
    barmode="group",
    title="Bias Metric Comparison (lower = fairer for SPD & MI; higher DI = fairer)",
    yaxis_title="Metric Value",
    height=420,
    legend=dict(orientation="h", y=1.1),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig_bar.add_hline(
    y=0.0, line_dash="dot", line_color="gray", annotation_text="Ideal (0.0) for SPD/MI"
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ── 4.3 Subgroup Distribution Comparison ──────────────────────────────────────
st.subheader("👥 Subgroup Distribution Comparison")
st.caption(
    "Compares how each subgroup is represented in Dataset A vs B. "
    "Large shifts can indicate sampling bias or population drift."
)

# Normalise counts to proportions for fair comparison across different dataset sizes
prop_a = (counts_a / counts_a.sum()).rename("Dataset A (%)")
prop_b = (counts_b / counts_b.sum()).rename("Dataset B (%)")
all_groups = sorted(set(prop_a.index.tolist()) | set(prop_b.index.tolist()))

prop_a = prop_a.reindex(all_groups, fill_value=0)
prop_b = prop_b.reindex(all_groups, fill_value=0)
dist_df = pd.DataFrame({"Dataset A (%)": prop_a * 100, "Dataset B (%)": prop_b * 100})

fig_dist = go.Figure()
fig_dist.add_trace(go.Bar(
    name="Dataset A",
    x=dist_df.index.astype(str),
    y=dist_df["Dataset A (%)"],
    marker_color="#2E86C1",
    opacity=0.9,
))
fig_dist.add_trace(go.Bar(
    name="Dataset B",
    x=dist_df.index.astype(str),
    y=dist_df["Dataset B (%)"],
    marker_color="#E74C3C",
    opacity=0.9,
))
fig_dist.update_layout(
    barmode="group",
    title="Subgroup Proportions (%) — Representation Shift",
    xaxis_title="Subgroup",
    yaxis_title="Proportion (%)",
    height=430,
    xaxis_tickangle=-35,
    legend=dict(orientation="h", y=1.1),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_dist, use_container_width=True)

# Show raw counts table
with st.expander("🔢 View raw subgroup counts"):
    raw_df = pd.DataFrame({
        "Dataset A — Count": counts_a.reindex(all_groups, fill_value=0),
        "Dataset A — %": (prop_a * 100).round(2),
        "Dataset B — Count": counts_b.reindex(all_groups, fill_value=0),
        "Dataset B — %": (prop_b * 100).round(2),
        "Proportion Shift (pp)": ((prop_b - prop_a) * 100).round(2),
    })
    raw_df.index.name = "Subgroup"
    st.dataframe(raw_df, use_container_width=True)

st.markdown("---")

# ── 4.4 Outcome Rate Comparison per Subgroup ──────────────────────────────────
st.subheader("🎯 Positive Outcome Rate per Subgroup")
st.caption(
    "Shows the proportion of positive outcomes (Y=1) for each subgroup in both datasets. "
    "Large differences signal distribution shift or label shift."
)

def _outcome_rates(df: pd.DataFrame, audit_col: str, target_col: str):
    try:
        tmp = df.copy()
        if not pd.api.types.is_numeric_dtype(tmp[target_col]):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            tmp[target_col] = le.fit_transform(tmp[target_col].astype(str))
        rates = tmp.groupby(audit_col)[target_col].mean()
        return rates
    except Exception:
        return pd.Series(dtype=float)

rates_a = _outcome_rates(
    R["df_a"] if len(R["sensitive_a"]) == 1 else R["df_a"].assign(
        intersectional_feature=create_intersectional_feature(R["df_a"], R["sensitive_a"])
    ),
    R["col_a"], R["target_a"]
)
rates_b = _outcome_rates(
    R["df_b"] if len(R["sensitive_b"]) == 1 else R["df_b"].assign(
        intersectional_feature=create_intersectional_feature(R["df_b"], R["sensitive_b"])
    ),
    R["col_b"], R["target_b"]
)

all_rate_groups = sorted(set(rates_a.index.astype(str)) | set(rates_b.index.astype(str)))
rates_a.index = rates_a.index.astype(str)
rates_b.index = rates_b.index.astype(str)
rates_a = rates_a.reindex(all_rate_groups, fill_value=np.nan)
rates_b = rates_b.reindex(all_rate_groups, fill_value=np.nan)

fig_rates = go.Figure()
fig_rates.add_trace(go.Bar(
    name="Dataset A",
    x=all_rate_groups,
    y=rates_a.values,
    marker_color="#2E86C1",
    text=[f"{v:.2%}" if not np.isnan(v) else "N/A" for v in rates_a.values],
    textposition="outside",
))
fig_rates.add_trace(go.Bar(
    name="Dataset B",
    x=all_rate_groups,
    y=rates_b.values,
    marker_color="#E74C3C",
    text=[f"{v:.2%}" if not np.isnan(v) else "N/A" for v in rates_b.values],
    textposition="outside",
))
fig_rates.update_layout(
    barmode="group",
    title="Subgroup Positive Outcome Rates",
    xaxis_title="Subgroup",
    yaxis_title="Positive Outcome Rate",
    yaxis_tickformat=".0%",
    height=430,
    xaxis_tickangle=-35,
    legend=dict(orientation="h", y=1.1),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_rates, use_container_width=True)

st.markdown("---")

# ── 4.5 Radar / Spider Chart — Multi-metric Overview ──────────────────────────
st.subheader("🕸️ Fairness Radar Chart")
st.caption("Spider chart normalizing all metrics to [0, 1] so you can see the overall fairness profile at a glance.")

def _safe(v, substitute=0.0):
    return float(v) if v is not None and not np.isnan(v) else substitute

# Normalise metrics to [0, 1] where 1 = perfectly fair
spd_norm_a = max(0.0, 1.0 - _safe(spd_a))
spd_norm_b = max(0.0, 1.0 - _safe(spd_b))
di_norm_a  = min(1.0, _safe(di_a))
di_norm_b  = min(1.0, _safe(di_b))
mi_norm_a  = max(0.0, 1.0 - min(1.0, _safe(mi_a)))
mi_norm_b  = max(0.0, 1.0 - min(1.0, _safe(mi_b)))

radar_categories = ["SPD Fairness", "DI Fairness", "MI Fairness"]
vals_a = [spd_norm_a, di_norm_a, mi_norm_a]
vals_b = [spd_norm_b, di_norm_b, mi_norm_b]

fig_radar = go.Figure()
for label, vals, color in [("Dataset A", vals_a, "#2E86C1"), ("Dataset B", vals_b, "#E74C3C")]:
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=radar_categories + [radar_categories[0]],
        fill="toself",
        name=label,
        line_color=color,
        opacity=0.6,
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    title="Fairness Profile Radar (1.0 = perfectly fair)",
    height=450,
)
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ── 4.6 Bias Drift Verdict ────────────────────────────────────────────────────
st.subheader("⚖️ Bias Drift Verdict")

issues, improvements = [], []
if spd_a is not None and spd_b is not None:
    delta_spd = spd_b - spd_a
    if delta_spd > 0.05:
        issues.append(f"SPD increased by **{delta_spd:+.4f}** — statistical parity is worse in Dataset B.")
    elif delta_spd < -0.05:
        improvements.append(f"SPD decreased by **{delta_spd:+.4f}** — Dataset B is more statistically fair.")

if di_a is not None and di_b is not None:
    delta_di = di_b - di_a
    if delta_di < -0.05:
        issues.append(f"DI dropped by **{delta_di:+.4f}** — disparate impact is worse in Dataset B.")
    elif delta_di > 0.05:
        improvements.append(f"DI rose by **{delta_di:+.4f}** — Dataset B has less disparate impact.")
    if di_b is not None and di_b < 0.8:
        issues.append(f"Dataset B DI = **{di_b:.4f}** — below the legal 80% threshold ⚠️")

if mi_a is not None and mi_b is not None:
    delta_mi = mi_b - mi_a
    if delta_mi > 0.02:
        issues.append(f"Mutual Information increased by **{delta_mi:+.4f}** — sensitive feature more correlated with outcome in Dataset B.")
    elif delta_mi < -0.02:
        improvements.append(f"Mutual Information decreased by **{delta_mi:+.4f}** — less proxy correlation in Dataset B.")

col_v1, col_v2 = st.columns(2)
with col_v1:
    if issues:
        st.error("🔴 Bias Issues Detected in Dataset B")
        for issue in issues:
            st.markdown(f"- {issue}")
    else:
        st.success("🟢 No significant bias regression detected in Dataset B.")

with col_v2:
    if improvements:
        st.success("✅ Fairness Improvements in Dataset B")
        for imp in improvements:
            st.markdown(f"- {imp}")
    else:
        st.info("ℹ️ No notable fairness improvements detected in Dataset B.")

# Summary table
summary_data = {
    "Metric": ["SPD", "DI", "MI"],
    "Dataset A": [_fmt(spd_a), _fmt(di_a), _fmt(mi_a)],
    "Dataset B": [_fmt(spd_b), _fmt(di_b), _fmt(mi_b)],
    "Drift": [
        _delta_str(spd_a, spd_b, lower_is_better=True),
        _delta_str(di_a, di_b, lower_is_better=False),
        _delta_str(mi_a, mi_b, lower_is_better=True),
    ],
    "Fair Threshold": ["< 0.1 (ideal)", "≥ 0.80 (legal)", "Near 0 (ideal)"],
}
st.subheader("📋 Summary Table")
st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "BiasScope Dataset Comparison Mode | "
    "Powered by Fairlearn, SHAP, and Scikit-Learn."
)
