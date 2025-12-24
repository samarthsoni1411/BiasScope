# pages/analyze.py
import streamlit as st
from modules.data_utils import read_dataset, guess_target_column

st.title("🔍 Analyze Dataset")
path = st.session_state.get("dataset_path", "")
if not path:
    st.warning("Please upload a dataset first.")
else:
    df = read_dataset(path)
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    target_guess = guess_target_column(df)
    task = "unsupervised" if not target_guess else "supervised"
    st.write(f"**Detected task:** {task}")
    if target_guess:
        nuniq = df[target_guess].nunique()
        if nuniq > 20:
            st.write(f"**Likely regression** (target: `{target_guess}`)")
        else:
            st.write(f"**Likely classification** (target: `{target_guess}`)")
    st.session_state["target_col"] = target_guess
    st.dataframe(df.head(5))
