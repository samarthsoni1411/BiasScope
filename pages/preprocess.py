# pages/preprocess.py
import streamlit as st
import pandas as pd
from modules.data_utils import read_dataset, preprocess_data

st.title("🧹 Preprocess Data")
path = st.session_state.get("dataset_path", "")
target_col = st.session_state.get("target_col", "")

if not path:
    st.warning("Please upload and analyze dataset first.")
    st.stop()

df = read_dataset(path)
st.write(f"Dataset loaded: {path.split('/')[-1]} — shape: {df.shape}")
st.dataframe(df.head(5))

if st.button("🚀 Run Preprocessing"):
    cleaned_path, df_clean, encoded_cols = preprocess_data(df, target_col)
    st.success(f"✅ Preprocessing complete. Saved to: {cleaned_path}")
    st.write("**Encoded columns:**", encoded_cols if encoded_cols else "None")
    st.dataframe(df_clean.head(5))
    st.session_state["cleaned_path"] = cleaned_path
