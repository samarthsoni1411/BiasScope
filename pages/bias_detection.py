# pages/bias_detection.py
import streamlit as st
import pandas as pd
from modules.data_utils import read_dataset, guess_target_column
from modules.bias_utils import calc_spd, calc_di, calc_mutual_info

st.title("⚖️ Data-Level Bias Detection")

path = st.session_state.get("dataset_path", "")
if not path:
    st.warning("Upload dataset first.")
    st.stop()

df = read_dataset(path)
target_col = st.session_state.get("target_col", "")
sensitive = st.selectbox("Choose Sensitive Feature", options=[c for c in df.columns if c != target_col])

st.write(f"Dataset: {path.split('/')[-1]} — shape: {df.shape}")

spd = calc_spd(df, sensitive, target_col) if target_col else None
di = calc_di(df, sensitive, target_col) if target_col else None
mi = calc_mutual_info(df, sensitive, target_col) if target_col else None

st.subheader("📊 Data-level metrics")
st.write(f"**Statistical Parity Difference (SPD):** {spd}")
st.write(f"**Disparate Impact (DI):** {di}")
st.write(f"**Mutual Information (sensitive, target):** {mi}")

st.markdown("**Note:** These metrics assume a binary target. For multi-class targets, consider domain-specific measures.")
