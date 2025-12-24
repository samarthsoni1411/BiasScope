# pages/report.py
import streamlit as st
import os
from modules.report_utils import generate_fairness_report

st.set_page_config(page_title="BiasScope | Report", layout="wide")
st.title("🧾 Generate Report")

cleaned_path = st.session_state.get("cleaned_path", "")
trained_model = st.session_state.get("trained_model", {})
fairness_before = st.session_state.get("model_fairness_results", {})
fairness_after = st.session_state.get("mitigated_fairness", {})

if not cleaned_path or not trained_model or not fairness_before or not fairness_after:
    st.warning("Complete mitigation before generating report.")
    st.stop()

dataset_name = os.path.basename(cleaned_path)
target_col = st.session_state.get("target_col", "")
sensitive_feature = st.session_state.get("sensitive_col", "Not Specified")
model_name = trained_model.get("best_model", "Unknown")

accuracy_before = trained_model.get("metrics", {}).get("accuracy", 0.0)
accuracy_after = trained_model.get("metrics", {}).get("accuracy", trained_model.get("metrics", {}).get("r2", 0.0))

if st.button("📄 Generate PDF Report"):
    with st.spinner("Generating..."):
        reports_dir = os.path.join("data", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        output_file = os.path.join(reports_dir, f"BiasScope_Report_{model_name}.pdf")
        generate_fairness_report(
            output_path=output_file,
            dataset_name=dataset_name,
            target_col=target_col,
            sensitive_feature=sensitive_feature,
            model_name=model_name,
            fairness_before=fairness_before,
            fairness_after=fairness_after,
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after
        )
        st.success("Report created.")
        with open(output_file, "rb") as f:
            st.download_button("⬇️ Download PDF", data=f, file_name=os.path.basename(output_file), mime="application/pdf")
