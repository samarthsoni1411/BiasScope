# pages/6_📈_Model_Bias.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from modules.data_utils import read_dataset
from modules.fairness_metrics import compute_model_fairness
from modules.bias_utils import create_intersectional_feature

st.set_page_config(page_title="BiasScope | Model Bias Detection", layout="wide")
st.title("📈 Model-Level Intersectional Bias Detection")

# -----------------------------
# 1. Load session state
# -----------------------------
model_info = st.session_state.get("trained_model", None)
cleaned_path = st.session_state.get("cleaned_path", "")
raw_path = st.session_state.get("dataset_path", "")

if not model_info or not cleaned_path or not raw_path:
    st.warning("⚠️ Please complete the previous steps (Upload & Train) first.")
    st.stop()

# Load both datasets: Scaled for prediction, Raw for human-readable labels
df_scaled = read_dataset(cleaned_path)
df_raw = read_dataset(raw_path)

model_path = model_info.get("model_path")
target_col = st.session_state.get("target_col", "")

# -----------------------------
# 2. Intersectional Audit Configuration
# -----------------------------
st.sidebar.header("Audit Configuration")
sensitive_cols = st.sidebar.multiselect(
    "Select Sensitive Features (Original Labels)", 
    options=[c for c in df_raw.columns if c != target_col],
    help="Select multiple features to perform an intersectional model audit with readable names."
)

min_n = st.sidebar.slider(
    "Minimum Group Size (n_min)", 
    5, 100, 20, 
    help="Groups smaller than this will be ignored to ensure statistical significance."
)

if not sensitive_cols:
    st.info("💡 Select one or more sensitive features from the sidebar to begin.")
    st.stop()

# Load model wrapper
try:
    with open(model_path, "rb") as f:
        wrapper = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model wrapper: {e}")
    st.stop()

model = wrapper.get("model") if isinstance(wrapper, dict) else wrapper
feature_maps = wrapper.get("feature_maps", {}) if isinstance(wrapper, dict) else {}
feature_cols = wrapper.get("feature_cols", None) if isinstance(wrapper, dict) else None

unique_targets = df_raw[target_col].dropna().unique().tolist()
pos_class = st.selectbox(
    "Choose positive class (favorable outcome)",
    options=unique_targets,
    index=1 if len(unique_targets) == 2 else 0
)

# -----------------------------
# 3. Run Bias Evaluation
# -----------------------------
if st.button("🚀 Run Intersectional Model Audit"):
    with st.spinner("Mapping labels and evaluating model fairness..."):
        
        # --- LOGIC STREAM A: Human-Readable Labels ---
        if len(sensitive_cols) > 1:
            audit_col_name = "intersectional_feature"
            display_labels = create_intersectional_feature(df_raw, sensitive_cols)
        else:
            audit_col_name = sensitive_cols[0]
            display_labels = df_raw[audit_col_name].astype(str)

        # Filter by statistical significance using Display Labels
        counts = display_labels.value_counts()
        valid_groups = counts[counts >= min_n].index.tolist()
        
        if len(valid_groups) < 2:
            st.error(f"Not enough subgroups meet the {min_n} threshold. Lower the threshold.")
            st.stop()
            
        # Synchronize indices across Scaled Data and Raw Labels
        mask = display_labels.isin(valid_groups)
        audit_df_scaled = df_scaled[mask].reset_index(drop=True)
        audit_labels = display_labels[mask].reset_index(drop=True)

        # --- LOGIC STREAM B: Prediction (Scaled Data) ---
        X_raw_input = audit_df_scaled.drop(columns=[target_col]).copy()
        y_true = (audit_df_scaled[target_col].astype(str) == str(pos_class)).astype(int)

        # Apply feature encoding alignment
        if feature_maps and feature_cols:
            X_enc = X_raw_input.copy()
            for col, fmap in feature_maps.items():
                if col not in X_enc.columns:
                    X_enc[col] = -1
                    continue
                if fmap is None:
                    try: X_enc[col] = pd.to_numeric(X_enc[col])
                    except: X_enc[col] = pd.factorize(X_enc[col].astype(str))[0]
                else:
                    X_enc[col] = X_enc[col].astype(str).map(fmap).fillna(-1).astype(int)
            X_pred = X_enc.reindex(columns=feature_cols, fill_value=-1)
        else:
            X_pred = X_raw_input

        # Model Prediction
        try:
            y_pred = model.predict(X_pred)
            if set(np.unique(y_pred)) - {0, 1}:
                y_pred = (pd.Series(y_pred).astype(str) == str(pos_class)).astype(int)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        y_pred = pd.Series(y_pred).reset_index(drop=True)

        # --- LOGIC STREAM C: Unified Results ---
        accuracy = (y_true == y_pred).mean()

        # Compute Fairness using Scaled Predictions and RAW Labels
        fairness_results = compute_model_fairness(
            y_true,
            y_pred,
            audit_labels # Using text labels for the audit table!
        )

        fairness_results["accuracy"] = accuracy
        st.session_state["model_fairness_results"] = fairness_results
        st.session_state["sensitive_col"] = f"Intersectional ({', '.join(sensitive_cols)})"

        st.success("Fairness evaluation complete.")

        # Summary Display
        st.subheader("📊 Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("DP Difference", round(fairness_results["Demographic Parity Difference"], 4))
        col2.metric("EO Difference", round(fairness_results["Equal Opportunity Difference"], 4))
        col3.metric("Overall Accuracy", round(accuracy, 4))

        # Group-wise breakdown (Human-Readable)
        group_df = pd.DataFrame({
            "Subgroup (Raw Labels)": list(fairness_results["Group Accuracy"].keys()),
            "Accuracy": list(fairness_results["Group Accuracy"].values()),
            "Recall": list(fairness_results["Group Recall"].values()),
            "Selection Rate": list(fairness_results["Group Selection Rate"].values())
        }).sort_values("Accuracy")

        st.subheader("👥 Performance by Intersectional Subgroup")
        st.dataframe(group_df, use_container_width=True)

        st.write("### Accuracy across Subgroups")
        st.bar_chart(group_df.set_index("Subgroup (Raw Labels)")["Accuracy"])