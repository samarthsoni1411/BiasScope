# pages/bias_detection.py
import streamlit as st
import pandas as pd
from modules.data_utils import read_dataset
from modules.bias_utils import calc_spd, calc_di, calc_mutual_info, create_intersectional_feature

st.title("⚖️ Data-Level Intersectional Bias Detection")

path = st.session_state.get("dataset_path", "")
if not path:
    st.warning("Please upload a dataset first.")
    st.stop()

df = read_dataset(path)
target_col = st.session_state.get("target_col", "")

# UI CONFIGURATION
st.sidebar.header("Audit Settings")
sensitive_cols = st.sidebar.multiselect(
    "Select Sensitive Features", 
    options=[c for c in df.columns if c != target_col],
    help="Select multiple features (e.g., Race + Gender) to perform an Intersectional Audit."
)

min_n = st.sidebar.slider(
    "Minimum Group Size (n_min)", 
    5, 100, 20, 
    help="Groups smaller than this will be ignored to maintain statistical significance."
)

if not sensitive_cols:
    st.info("💡 Select one or more sensitive features from the sidebar to begin.")
    st.stop()

# INTERSECTIONAL LOGIC
audit_col_name = "intersectional_feature"
if len(sensitive_cols) > 1:
    df[audit_col_name] = create_intersectional_feature(df, sensitive_cols)
    st.info(f"🧬 **Intersectional Audit Active:** Combining {', '.join(sensitive_cols)}")
else:
    audit_col_name = sensitive_cols[0]

# CALCULATION
spd = calc_spd(df, audit_col_name, target_col, min_samples=min_n)
di = calc_di(df, audit_col_name, target_col, min_samples=min_n)
mi = calc_mutual_info(df, audit_col_name, target_col)

# RESULTS DISPLAY
st.markdown("---")
st.subheader("📊 Key Metrics")
m1, m2, m3 = st.columns(3)
m1.metric("Statistical Parity Diff (SPD)", round(spd, 4) if spd is not None else "N/A")
m2.metric("Disparate Impact (DI)", round(di, 4) if di is not None else "N/A")
m3.metric("Mutual Information", round(mi, 4) if mi is not None else "N/A")

# DISTRIBUTION VISUALIZATION
st.write("### 🔍 Subgroup Distribution")
counts = df[audit_col_name].value_counts()
st.bar_chart(counts)

if spd is None or di is None:
    st.warning(f"⚠️ **Small Data Warning:** Not enough groups met the {min_n} sample threshold. Try lowering the threshold or selecting fewer features.")

st.markdown("""
> **Why Intersectionality?** A model can appear fair for Gender and Race individually but hide deep biases against specific subgroups (e.g., *Black Women*). This audit checks the most granular slices of your data.
""")