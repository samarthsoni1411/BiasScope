# pages/7_🧬_Mitigation.py
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from modules.data_utils import read_dataset
from modules.fairness_metrics import compute_model_fairness
from modules.mitigation_utils import mitigate_bias_reweighing

st.set_page_config(page_title="BiasScope | Mitigation", layout="wide")
st.title("🧬 Bias Mitigation")

# -----------------------------
# Load required session state
# -----------------------------
cleaned_path = st.session_state.get("cleaned_path", "")
trained_model_info = st.session_state.get("trained_model", None)
fairness_before = st.session_state.get("model_fairness_results", None)

if not cleaned_path or not trained_model_info or not fairness_before:
    st.warning("Train model and run model-bias first.")
    st.stop()

df = read_dataset(cleaned_path)
target_col = st.session_state.get("target_col", "")
sensitive_options = [c for c in df.columns if c != target_col]

# -----------------------------
# Inputs
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    target = st.selectbox(
        "Target column",
        options=df.columns,
        index=df.columns.get_loc(target_col) if target_col in df.columns else 0
    )
with col2:
    sensitive = st.selectbox(
        "Sensitive attribute",
        options=sensitive_options
    )

# -----------------------------
# Run Mitigation
# -----------------------------
if st.button("🚀 Run Reweighing Mitigation"):
    with st.spinner("Running mitigation..."):
        result = mitigate_bias_reweighing(df, target, sensitive)
        if result.get("status") != "success":
            st.error(f"Mitigation failed: {result.get('message')}")
            st.stop()

        # Load mitigated model wrapper
        try:
            with open(result["model_path"], "rb") as f:
                wrapper = pickle.load(f)
        except Exception as e:
            st.error(f"Failed to load mitigated wrapper: {e}")
            st.stop()

        mitigated_model = wrapper.get("model")
        feature_maps = wrapper.get("feature_maps", {})
        feature_cols = wrapper.get("feature_cols", [])
        target_mapping = wrapper.get("target_mapping", None)

        acc_after = result.get("accuracy")
        st.success(f"Mitigation complete. Accuracy on held-out: {acc_after:.3f}")

        # -----------------------------
        # Prepare y_true
        # -----------------------------
        y_true = df[target].astype(str)
        if target_mapping:
            y_true = y_true.map(target_mapping)
        else:
            try:
                y_true = pd.to_numeric(y_true)
            except Exception:
                pos = y_true.mode().iloc[0]
                y_true = (y_true == pos).astype(int)

        # -----------------------------
        # Prepare X for prediction
        # -----------------------------
        X_raw = df.drop(columns=[target]).copy()
        for col, fmap in feature_maps.items():
            if col not in X_raw.columns:
                X_raw[col] = -1
                continue
            if fmap is None:
                try:
                    X_raw[col] = pd.to_numeric(X_raw[col])
                except Exception:
                    X_raw[col] = pd.factorize(X_raw[col].astype(str))[0]
            else:
                X_raw[col] = (
                    X_raw[col].astype(str)
                    .map(fmap)
                    .fillna(-1)
                    .astype(int)
                )

        X_pred = X_raw.reindex(columns=feature_cols, fill_value=-1)

        try:
            y_pred = mitigated_model.predict(X_pred)
        except Exception as e:
            st.error(f"Prediction failed on mitigated model: {e}")
            st.stop()

        y_pred = pd.Series(y_pred).reset_index(drop=True)

        # -----------------------------
        # Fairness after mitigation
        # -----------------------------
        fairness_after = compute_model_fairness(y_true, y_pred, df[sensitive])
        st.session_state["mitigated_fairness"] = fairness_after

        # -----------------------------
        # Before vs After Table
        # -----------------------------
        comp_df = pd.DataFrame({
            "Metric": [
                "Demographic Parity Difference",
                "Equal Opportunity Difference"
            ],
            "Before Mitigation": [
                fairness_before.get("Demographic Parity Difference"),
                fairness_before.get("Equal Opportunity Difference")
            ],
            "After Mitigation": [
                fairness_after.get("Demographic Parity Difference"),
                fairness_after.get("Equal Opportunity Difference")
            ]
        })

        st.subheader("📊 Before vs After Fairness Metrics")
        st.dataframe(comp_df)

        # ==============================
        # 📈 Fairness Before vs After Visualization
        # ==============================
        st.subheader("📈 Fairness Change Visualization")

        metrics = ["Demographic Parity Difference", "Equal Opportunity Difference"]
        before_vals = [
            fairness_before.get("Demographic Parity Difference"),
            fairness_before.get("Equal Opportunity Difference"),
            ]
        after_vals = [
            fairness_after.get("Demographic Parity Difference"),
            fairness_after.get("Equal Opportunity Difference"),
        ]
        fig = go.Figure()

        fig.add_bar(
            x=metrics,
            y=before_vals,
            name="Before Mitigation"
        )

        fig.add_bar(
            x=metrics,
            y=after_vals,
            name="After Mitigation"
        )

        fig.update_layout(
            barmode="group",
            xaxis_title="Fairness Metrics",
            yaxis_title="Metric Value",
            legend_title="Stage",
            height=420,
        )

        st.plotly_chart(fig, use_container_width=True)


        # ==============================
        # 🔎 Mitigation Interpretation
        # ==============================
        st.subheader("🔎 Mitigation Interpretation")

        dp_before = fairness_before.get("Demographic Parity Difference")
        eo_before = fairness_before.get("Equal Opportunity Difference")
        acc_before = fairness_before.get("accuracy", None)

        dp_after = fairness_after.get("Demographic Parity Difference")
        eo_after = fairness_after.get("Equal Opportunity Difference")

        delta_dp = dp_before - dp_after
        delta_eo = eo_before - eo_after
        delta_acc = acc_after - acc_before if acc_before is not None else None

        interpretation = []

        if delta_dp > 0:
            interpretation.append("Demographic parity improved after mitigation.")
        else:
            interpretation.append("Demographic parity worsened after mitigation.")

        if delta_eo > 0:
            interpretation.append("Equal opportunity improved after mitigation.")
        else:
            interpretation.append("Equal opportunity slightly worsened after mitigation.")

        if delta_acc is not None:
            if delta_acc < 0:
                interpretation.append(
                    f"Model accuracy decreased by {abs(delta_acc):.3f} after mitigation."
                )
            else:
                interpretation.append(
                    f"Model accuracy improved by {delta_acc:.3f} after mitigation."
                )

        st.markdown("**Interpretation:**")
        for line in interpretation:
            st.write(f"- {line}")

        # ==============================
        # ⚖️ Verdict
        # ==============================
        if delta_dp > 0 and (delta_acc is None or abs(delta_acc) <= 0.03):
            verdict = "🟢 Recommended – fairness improved with acceptable performance loss."
        elif delta_dp > 0 or delta_eo > 0:
            verdict = "🟡 Trade-off detected – mitigation requires review."
        else:
            verdict = "🔴 Not recommended – fairness did not improve."

        st.markdown(f"### Verdict: {verdict}")
