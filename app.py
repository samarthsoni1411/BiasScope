# app.py
import streamlit as st

st.set_page_config(page_title="BiasScope", layout="wide", page_icon="⚖️")
st.title("⚖️ BiasScope — Fairness Detection & Mitigation Framework")
st.markdown("""
### Welcome to BiasScope!
Use the sidebar to navigate:
1. 📂 Upload dataset
2. 🔍 Analyze dataset
3. ⚖️ Data-level bias detection
4. 🧹 Preprocess data
5. 🤖 Train model
6. 📈 Model bias detection
7. 🧬 Mitigation
8. 🧾 Report
""")
st.sidebar.info("Open pages from the app's sidebar (if using multipage Streamlit, place each page file in `pages/`).")
