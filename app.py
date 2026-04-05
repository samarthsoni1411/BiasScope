# app.py
import streamlit as st

st.set_page_config(
    page_title="BiasScope",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hero Section
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>⚖️ BiasScope</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray;'>An Interactive Framework for AI Fairness Discovery & Mitigation</h3>", unsafe_allow_html=True)
st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 🚀 Welcome to BiasScope!
    BiasScope is a comprehensive, open-source tool designed to help researchers and practitioners audit, understand, and mitigate bias in machine learning models. 
    
    Rather than just treating fairness as a single binary check, BiasScope enables **intersectional bias discovery**, automated **Pareto-frontier mitigation**, and deep **model explainability**.
    """)
    
    st.info("👈 **Use the sidebar to navigate through the AI fairness pipeline.**")

with col2:
    st.markdown("""
    ### 📌 Core Capabilities
    - 🔍 **Intersectional Audits**: Find hidden biases across combined subgroups.
    - 🧬 **Multi-Objective Mitigation**: Explore trade-offs between Accuracy and Demographic Parity.
    - 🧠 **Explainability**: Understand model decisions with SHAP feature importance.
    - 🔄 **Counterfactuals**: Test real-time "What-If" scenarios.
    - 🔀 **Dataset Comparison**: Compare bias drift between two datasets side by side.
    - 🧾 **Reporting**: Generate automated PDF audit reports.
    """)

st.markdown("---")

# Pipeline Visual Workflow
st.markdown("### 🛠️ The BiasScope Workflow")
flow1, flow2, flow3, flow4, flow5 = st.columns(5)

with flow1:
    st.success("**1. Data Auditing**")
    st.write("Upload your dataset and immediately identify systemic data-level biases.")

with flow2:
    st.warning("**2. Model Training**")
    st.write("Automatically train an ensemble of fast ML models on your uploaded data.")

with flow3:
    st.info("**3. Bias Explanation**")
    st.write("Calculate model-level disparities and explain them using SHAP.")

with flow4:
    st.error("**4. Mitigation & Reports**")
    st.write("Balance fairness against accuracy and generate a final PDF audit report.")

with flow5:
    st.success("**5. Dataset Comparison**")
    st.write("Compare bias drift across two datasets — train vs. live, before vs. after collection.")

st.sidebar.success("BiasScope Core Pipeline ready. Select a page above to begin.")
