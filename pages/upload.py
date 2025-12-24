# pages/upload.py
import streamlit as st
from modules.data_utils import save_uploaded_file, read_dataset

st.title("📂 Upload Dataset")
file = st.file_uploader("Upload dataset (.csv, .xlsx)", type=["csv","xlsx"])
if file:
    path = save_uploaded_file(file)
    st.success(f"✅ File saved at: {path}")
    df = read_dataset(path)
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head(5))
    st.session_state["dataset_path"] = path
