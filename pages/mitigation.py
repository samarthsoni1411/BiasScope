import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.data_utils import read_dataset
from modules.fairness_metrics import compute_model_fairness
from modules.mitigation_utils import mitigate_bias_reweighing, mitigate_bias_grid_search, cleanup_repaired_dir
from modules.model_utils import predict_with_model

st.set_page_config(page_title="BiasScope | Mitigation", layout="wide")
st.title("🧬 Bias Mitigation")

# FIX 5: Sidebar disk-cleanup utility for accumulated GridSearch .pkl files
with st.sidebar.expander("🗑️ Storage Management"):
    num_files = len([f for f in __import__("os").listdir(
        __import__("os").path.join("data", "repaired")
    ) if f.endswith(".pkl")]) if __import__("os").path.exists(__import__("os").path.join("data", "repaired")) else 0
    st.write(f"Repaired models on disk: **{num_files}** file(s)")
    if st.button("🧹 Clean old repaired models (keep 5 newest)"):
        deleted = cleanup_repaired_dir(keep_newest=5)
        st.success(f"Deleted {deleted} old model file(s).")

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

st.markdown("---")
mitigation_type = st.radio(
    "Mitigation Strategy:", 
    ["⚡ Standard Optimization (Exponentiated Gradient)", "📈 Trade-off Analysis (GridSearch Pareto Frontier)"],
    help="Optimization finds one best model. Trade-off Analysis explores the Accuracy vs Fairness spectrum."
)

if st.button("🚀 Run Mitigation"):
    if "Standard" in mitigation_type:
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
                
                # Make sure report gets the mitigated model Wrapper for prediction
                st.session_state["mitigated_model_wrapper"] = wrapper
                
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
                st.error(result.get("message", "Unknown Error"))
                
    else:
        with st.spinner("Running GridSearch to find Trade-off Frontier..."):
            st.session_state["grid_result"] = mitigate_bias_grid_search(df, target, sensitive, grid_size=15)

if "grid_result" in st.session_state and "Trade-off" in mitigation_type:
    res = st.session_state["grid_result"]
    if res["status"] != "success":
        st.error(res.get("message", "Error in GridSearch"))
    else:
        models = res["models"]
        df_models = pd.DataFrame(models)
        
        st.success("✅ Trade-off Models Generated Successfully!")
        st.markdown("### Accuracy vs. Disparity Trade-off Frontier")
        st.write("Each point represents an automated AI model evaluating a specific constraint weight.")
        
        # Plotly Scatter
        fig = px.scatter(
            df_models, 
            x="dp_diff", 
            y="accuracy", 
            text="model_id",
            labels={"dp_diff": "Demographic Parity Difference (Lower is Fairer)", "accuracy": "Accuracy (Higher is Better)"},
            title="Pareto Frontier (DP Diff vs Accuracy)"
        )
        fig.update_traces(textposition='top center', marker=dict(size=12, color='royalblue'))
        
        # Add original model as baseline
        orig_acc = trained_info.get("results", [{}])[0].get("Accuracy", 0.8) # rough fallback
        for r in trained_info.get("results", []):
            if r.get("Model") == trained_info.get("best_model"):
                orig_acc = r.get("Accuracy", orig_acc)
                
        orig_dp = fair_before.get("Demographic Parity Difference", 1.0)
        fig.add_scatter(x=[orig_dp], y=[orig_acc], mode='markers+text', text=['Original Model'], 
                        marker=dict(size=15, color='red', symbol='star'), name='Original Baseline')
                        
        st.plotly_chart(fig, use_container_width=True)
        
        # Select best model
        st.markdown("### Choose Optimal Model")
        selected_id = st.selectbox("Select Model ID from Trade-off Curve", options=df_models["model_id"].tolist(), index=0)
        
        selected_model_data = df_models[df_models["model_id"] == selected_id].iloc[0]
        
        st.info(f"**Selected Model [{selected_id}]** → Accuracy: {selected_model_data['accuracy']:.4f}  |  Demographic Parity Diff: {selected_model_data['dp_diff']:.4f}")
        
        if st.button("💾 Save Selected Model & Update Metrics"):
            wrapper = selected_model_data["wrapper"]
            y_true = df[target].astype(str).map(res["label_mapping"]).astype(int)
            y_pred = predict_with_model(wrapper, df)
            
            fair_after = compute_model_fairness(y_true, y_pred, df[sensitive])
            st.session_state["mitigated_fairness"] = fair_after
            st.session_state["mitigated_model_wrapper"] = wrapper
            st.success("Selected model saved for your final report!")