# pages/6_📈_Model_Bias.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from modules.data_utils import read_dataset
from modules.fairness_metrics import compute_model_fairness

st.set_page_config(page_title="BiasScope | Model Bias Detection", layout="wide")
st.title("📈 Model-Level Bias Detection")

# -----------------------------
# Load session state
# -----------------------------
model_info = st.session_state.get("trained_model", None)
cleaned_path = st.session_state.get("cleaned_path", "")

if not model_info or not cleaned_path:
    st.warning("Please train a model first.")
    st.stop()

df = read_dataset(cleaned_path)
model_path = model_info.get("model_path")
target_col = st.session_state.get("target_col", "")

st.markdown(f"**Loaded Model:** `{model_info.get('best_model','N/A')}`")
st.markdown(f"**Dataset:** `{cleaned_path.split('/')[-1]}` — shape: {df.shape}")

# -----------------------------
# Load model wrapper
# -----------------------------
try:
    with open(model_path, "rb") as f:
        wrapper = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model wrapper: {e}")
    st.stop()

model = wrapper.get("model") if isinstance(wrapper, dict) else wrapper
feature_maps = wrapper.get("feature_maps", {}) if isinstance(wrapper, dict) else {}
feature_cols = wrapper.get("feature_cols", None) if isinstance(wrapper, dict) else None

# -----------------------------
# User Inputs
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
        options=[c for c in df.columns if c != target]
    )

unique_targets = df[target].dropna().unique().tolist()
pos_class = None
if len(unique_targets) == 2:
    pos_class = st.selectbox(
        "Choose positive class",
        options=unique_targets,
        index=1
    )

# -----------------------------
# Run Bias Evaluation
# -----------------------------
if st.button("🚀 Run Model Bias Evaluation"):
    with st.spinner("Evaluating model fairness..."):
        X_raw = df.drop(columns=[target]).copy()
        y_true = df[target].reset_index(drop=True)

        # Apply feature encoding
        if feature_maps and feature_cols:
            X_enc = X_raw.copy()
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
            X_pred = X_enc.reindex(columns=feature_cols, fill_value=-1)
        else:
            X_pred = X_raw.copy()
            for c in X_pred.columns:
                if not pd.api.types.is_numeric_dtype(X_pred[c]):
                    X_pred[c] = pd.factorize(X_pred[c].astype(str))[0]

        # Prediction
        try:
            y_pred = model.predict(X_pred)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        y_pred = pd.Series(y_pred).reset_index(drop=True)

        # Binary target enforcement
        if len(unique_targets) != 2:
            st.error("Target must be binary for fairness metrics.")
            st.stop()

        neg_class = [v for v in unique_targets if v != pos_class][0]
        mapping = {str(neg_class): 0, str(pos_class): 1}

        # Map y_true using selected positive class
        y_true = y_true.astype(str)
        y_true = (y_true == str(pos_class)).astype(int)

        # Ensure y_pred is binary
        if y_pred.dtype != int and y_pred.dtype != bool:
            y_pred = pd.to_numeric(y_pred, errors="coerce")

        # If model outputs labels instead of 0/1
        if set(y_pred.dropna().unique()) - {0,1}:
            y_pred = (y_pred.astype(str) == str(pos_class)).astype(int)

        # Accuracy
        accuracy = (y_true == y_pred).mean()

        # ✅ FIXED FAIRNESS CALL (POSITIONAL ARGUMENTS)
        fairness_results = compute_model_fairness(
            y_true,
            y_pred,
            df[sensitive]
        )

        fairness_results["accuracy"] = accuracy
        st.session_state["model_fairness_results"] = fairness_results

        st.success("Fairness evaluation complete.")

        # Fairness summary
        st.subheader("📊 Fairness Summary")
        st.table(pd.DataFrame({
            "Metric": [
                "Demographic Parity Difference",
                "Equal Opportunity Difference",
                "Accuracy"
            ],
            "Value": [
                fairness_results["Demographic Parity Difference"],
                fairness_results["Equal Opportunity Difference"],
                accuracy
            ]
        }).set_index("Metric"))

        # Bias interpretation
        dp = fairness_results["Demographic Parity Difference"]
        if abs(dp) > 0.10:
            st.warning("⚠️ High demographic parity disparity detected.")
        elif abs(dp) > 0.05:
            st.info("ℹ️ Moderate demographic parity disparity detected.")
        else:
            st.success("✅ Low demographic parity disparity detected.")

        st.markdown(
            "📌 **Interpretation:** Differences in selection rates and recall across sensitive groups "
            "indicate potential bias in model predictions."
        )

        # Group-wise breakdown
        group_df = pd.DataFrame({
            "Group": list(fairness_results["Group Accuracy"].keys()),
            "Accuracy": list(fairness_results["Group Accuracy"].values()),
            "Recall": list(fairness_results["Group Recall"].values()),
            "Selection Rate": list(fairness_results["Group Selection Rate"].values())
        })

        st.subheader("👥 Group-wise Performance")
        st.dataframe(group_df)
