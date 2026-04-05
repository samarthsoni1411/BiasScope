import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from modules.data_utils import read_dataset

st.set_page_config(page_title="BiasScope | Explainability", layout="wide")
st.title("🧠 Model Explainability (SHAP & What-If)")

# -----------------------------
# 1. Load session state
# -----------------------------
model_info = st.session_state.get("trained_model")
cleaned_path = st.session_state.get("cleaned_path")
raw_path = st.session_state.get("dataset_path")
scaler_path = st.session_state.get("scaler_path")  # FIX 1: load saved scaler path

if not model_info or not cleaned_path or not raw_path:
    st.warning("⚠️ Please complete the Upload and Train steps first.")
    st.stop()

df_scaled = read_dataset(cleaned_path)
df_raw = read_dataset(raw_path)
target_col = st.session_state.get("target_col", "")

model_path = model_info.get("model_path")
try:
    with open(model_path, "rb") as f:
        wrapper = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

model = wrapper.get("model") if isinstance(wrapper, dict) else wrapper
feature_maps = wrapper.get("feature_maps", {}) if isinstance(wrapper, dict) else {}
feature_cols = wrapper.get("feature_cols", None) if isinstance(wrapper, dict) else None

# FIX 1: Load the saved StandardScaler (if available) for correct counterfactual predictions
_scaler_obj = None
_scaler_numeric_cols = []
if scaler_path:
    try:
        with open(scaler_path, "rb") as _sf:
            _scaler_bundle = pickle.load(_sf)
        _scaler_obj = _scaler_bundle.get("scaler")
        _scaler_numeric_cols = _scaler_bundle.get("numeric_cols", [])
    except Exception:
        _scaler_obj = None  # Graceful fallback if scaler file is missing

# Prepare Prediction Data
X_raw_input = df_scaled.drop(columns=[target_col]).copy()
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

st.markdown("---")

tab1, tab2 = st.tabs(["🌎 Global Explanations (SHAP)", "🔄 What-If Playground (Counterfactuals)"])

with tab1:
    st.header("Global Feature Importance")
    st.write("Understand which features drive the model's predictions overall.")
    
    if st.button("Generate SHAP Explanations"):
        with st.spinner("Calculating SHAP values (this may take a moment)..."):
            try:
                # Use a background sample for tree/kernel explainers to speed it up
                sample_size = min(100, len(X_pred))
                background = shap.sample(X_pred, sample_size)
                
                # Explainer selection
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_pred.sample(min(500, len(X_pred))))
                except:
                    explainer = shap.Explainer(model.predict, background)
                    shap_values = explainer(X_pred.sample(min(500, len(X_pred))))
                
                st.success("SHAP values calculated successfully.")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values[1], X_pred.sample(min(500, len(X_pred))), show=False)
                else:
                    shap.summary_plot(shap_values, X_pred.sample(min(500, len(X_pred))), show=False)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Could not generate SHAP values for this model type: {e}")

with tab2:
    st.header("What-If Playground")
    st.write("Select a single instance and tweak its features to see how the prediction changes.")
    
    # Instance Selection
    row_idx = st.number_input("Select Row Index:", min_value=0, max_value=len(df_raw)-1, value=0, step=1)
    
    st.subheader("Original Data")
    original_row = df_raw.iloc[[row_idx]].drop(columns=[target_col])
    st.dataframe(original_row)
    
    # Prediction of Original
    orig_pred_row = X_pred.iloc[[row_idx]]
    orig_pred = model.predict(orig_pred_row)[0]
    st.info(f"**Original Prediction (Encoded Label):** {orig_pred}")
    
    st.markdown("---")
    st.subheader("Tweak Features")
    
    # Create dynamic inputs for the user to change features
    tweak_dict = {}
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(original_row.columns):
        val = original_row[col].iloc[0]

        
        # Display inputs staggered in two columns
        target_col_ui = col1 if i % 2 == 0 else col2
        
        if pd.api.types.is_numeric_dtype(df_raw[col]):
            tweak_dict[col] = target_col_ui.number_input(f"{col}", value=float(val))
        else:
            unique_vals = df_raw[col].dropna().unique().tolist()
            # Handle potential type mismatches in unique values list
            try:
                default_idx = unique_vals.index(val)
            except ValueError:
                unique_vals.insert(0, val)
                default_idx = 0
            tweak_dict[col] = target_col_ui.selectbox(f"{col}", options=unique_vals, index=default_idx)
            
    if st.button("🔄 Predict with Tweaked Features"):
        tweaked_df = pd.DataFrame([tweak_dict])
        tweaked_enc = tweaked_df.copy()

        if feature_maps and feature_cols:
            for col, fmap in feature_maps.items():
                if col not in tweaked_enc.columns:
                    tweaked_enc[col] = -1
                    continue
                if fmap is None:  # Numeric column
                    try:
                        tweaked_enc[col] = pd.to_numeric(tweaked_enc[col])
                    except Exception:
                        tweaked_enc[col] = pd.factorize(tweaked_enc[col].astype(str))[0]
                else:  # Categorical column
                    tweaked_enc[col] = tweaked_enc[col].astype(str).map(fmap).fillna(-1).astype(int)

            tweaked_pred_ready = tweaked_enc.reindex(columns=feature_cols, fill_value=-1)
        else:
            tweaked_pred_ready = tweaked_enc

        # FIX 1: Apply the saved StandardScaler to numeric columns so prediction is correct
        if _scaler_obj is not None and _scaler_numeric_cols:
            cols_to_scale = [c for c in _scaler_numeric_cols if c in tweaked_pred_ready.columns]
            if cols_to_scale:
                try:
                    tweaked_pred_ready[cols_to_scale] = _scaler_obj.transform(
                        tweaked_pred_ready[cols_to_scale]
                    )
                except Exception:
                    st.warning("⚠️ Could not apply scaler to tweaked features — using raw values.")

        try:
            new_pred = model.predict(tweaked_pred_ready)[0]
            if new_pred != orig_pred:
                st.success(f"**New Prediction:** {new_pred} ✅ (Changed!)")
            else:
                st.warning(f"**New Prediction:** {new_pred} (Unchanged)")
        except Exception as e:
            st.error(f"Prediction error: {e}")
