BiasScope is an end-to-end Responsible AI system for detecting, interpreting, and mitigating bias in machine learning models trained on unknown datasets.
It supports model-level fairness auditing, group-wise analysis, and bias mitigation with trade-off evaluation, all through an interactive Streamlit interface.


🚀 Key Features
🔍 Model-Level Bias Detection
Supports trained ML models on unknown datasets

Computes fairness metrics:
Demographic Parity Difference
Equal Opportunity Difference

Displays group-wise performance:
Accuracy
Recall
Selection Rate
Automatic bias severity classification (Low / Moderate / High)
Human-readable interpretation of bias results


🧬 Bias Mitigation
Implements Reweighing (industry-accepted bias mitigation technique)
Retrains the best-performing model after mitigation
Compares Before vs After fairness metrics
Tracks accuracy–fairness trade-offs

Provides an automated verdict:
Recommended
Trade-off detected
Not recommended


📊 Interactive Visualizations
Before vs After fairness comparison
Clear tabular and graphical outputs
Designed for demos, interviews, and presentations


Project Architecture
BiasScope/
├── app.py                     # Main Streamlit app
├── pages/
│   ├── 6_📈_Model_Bias.py       # Model-level bias detection
│   └── 7_🧬_Mitigation.py       # Bias mitigation & evaluation
├── modules/
│   ├── data_utils.py
│   ├── fairness_metrics.py
│   └── mitigation_utils.py
├── README.md
├── .gitignore


Tech Stack :-
Python
Streamlit – interactive UI
Scikit-learn
CatBoost / Tree-based models
Pandas, NumPy
Plotly – visualizations


📈 Fairness Metrics Explained
Demographic Parity Difference (DPD)
Measures disparity in positive prediction rates across sensitive groups.
Equal Opportunity Difference (EOD)
Measures disparity in recall (true positive rate) across groups.
BiasScope not only computes these metrics but also explains their implications and flags risk severity.



🧪 How It Works (High Level)
Train a machine learning model on a dataset
Select:
Target variable
Sensitive attribute
Positive class
Run Model Bias Evaluation
Inspect fairness metrics and group-wise performance
Apply Bias Mitigation
Compare fairness and accuracy before vs after
Review automated interpretation and verdict


🎯 Use Cases
Fairness auditing of ML models
Responsible AI research and experimentation
Academic projects and demonstrations
Interview-ready portfolio project


📌 Project Status

✅ Model-level bias detection
✅ Bias mitigation with trade-off analysis
✅ Interpretation & decision support
🚧 Future work: additional mitigation techniques, reporting automation


👤 Author

Samarth Soni
B.Tech CSE | M.Tech AI/ML
Interested in Responsible AI, Fair ML, and Applied Machine Learning


⭐ Acknowledgement
Inspired by real-world Responsible AI practices and fairness evaluation frameworks.


