import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from modules.data_utils import read_dataset
from modules.fairness_metrics import compute_model_fairness
from modules.mitigation_utils import mitigate_bias_reweighing
from modules.model_utils import predict_with_model

st.set_page_config(page_title="BiasScope | Mitigation", layout="wide")
st.title("🧬 Bias Mitigation")

cleaned_path = st.session_state.get("cleaned_path")
trained_info = st.session_state.get("trained_model")
fair_before = st.session_state.get("model_fairness_results")

if not all([cleaned_path, trained_info, fair_before]):
    st.warning("Please complete Upload, Train, and Model-Bias steps first.")
    st.stop()

df = read_dataset(cleaned_path)
target_col = st.session_state.get("target_col")

col1, col2 = st.columns(2)
with col1:
    target = st.selectbox("Target", [target_col] if target_col in df.columns else df.columns)
with col2:
    sensitive = st.selectbox("Sensitive Attribute", [c for c in df.columns if c != target])

if st.button("🚀 Run Mitigation"):
    with st.spinner("Optimizing for fairness..."):
        result = mitigate_bias_reweighing(df, target, sensitive)
        
        if result["status"] == "success":
            import pickle
            with open(result["model_path"], "rb") as f:
                wrapper = pickle.load(f)
            
            y_true = df[target].astype(str).map(wrapper["target_mapping"]).astype(int)
            y_pred = predict_with_model(wrapper, df)
            
            fair_after = compute_model_fairness(y_true, y_pred, df[sensitive])
            st.session_state["mitigated_fairness"] = fair_after
            
            st.success(f"Mitigation Complete! Accuracy: {result['accuracy']:.3f}")
            
            metrics = ["Demographic Parity Difference", "Equal Opportunity Difference"]
            fig = go.Figure()
            fig.add_bar(x=metrics, y=[fair_before.get(m, 0) for m in metrics], name="Before")
            fig.add_bar(x=metrics, y=[fair_after.get(m, 0) for m in metrics], name="After")
            fig.update_layout(barmode="group", title="Fairness Improvement")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Comparison Table")
            comp = pd.DataFrame({
                "Metric": metrics,
                "Before": [fair_before.get(m, 0) for m in metrics],
                "After": [fair_after.get(m, 0) for m in metrics]
            })
            st.table(comp)
        else:
            st.error(result["message"])