import uuid
import streamlit as st
import os
from modules.report_utils import generate_fairness_report

st.set_page_config(page_title="BiasScope | Report", layout="wide")
st.title("🧾 Generate Fairness Audit Report")

cleaned_path = st.session_state.get("cleaned_path", "")
trained_model = st.session_state.get("trained_model", {})
fairness_before = st.session_state.get("model_fairness_results", {})
fairness_after = st.session_state.get("mitigated_fairness", {})

if not all([cleaned_path, trained_model, fairness_before, fairness_after]):
    st.warning("⚠️ Complete the full pipeline (Upload -> Train -> Bias Detection -> Mitigation) to generate a report.")
    st.stop()

dataset_name = os.path.basename(cleaned_path)
target_col = st.session_state.get("target_col", "Target")
sensitive_feature = st.session_state.get("sensitive_col", "Sensitive Attribute")
model_name = trained_model.get("best_model", "Best Model")

# Pull accuracy directly from the evaluation results
acc_before = 0.0
for res in trained_model.get("results", []):
    if res.get("Model") == model_name:
        acc_before = res.get("Accuracy", res.get("R2 Score", 0.0))
        break

# Mitigation accuracy comes from the session state calculated in mitigation.py
acc_after = fairness_after.get("accuracy", 0.0)

if st.button("📄 Generate PDF Audit Report"):
    with st.spinner("Compiling results into PDF..."):
        reports_dir = os.path.join("data", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        output_file = os.path.join(reports_dir, f"BiasScope_Audit_{uuid.uuid4().hex[:4]}.pdf")
        
        try:
            generate_fairness_report(
                output_path=output_file,
                dataset_name=dataset_name,
                target_col=target_col,
                sensitive_feature=sensitive_feature,
                model_name=model_name,
                fairness_before=fairness_before,
                fairness_after=fairness_after,
                accuracy_before=acc_before,
                accuracy_after=acc_after
            )
            
            st.success("✅ Report successfully generated!")
            with open(output_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=f,
                    file_name=os.path.basename(output_file),
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"Report generation failed: {e}")