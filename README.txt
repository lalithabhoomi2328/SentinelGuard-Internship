SENTINELGUARD – PROJECT 1 SUBMISSION PACKAGE

Student: Lalitha S
College: GSSS SSFGC FIRST GRADE DEGREE COLLEGE
Course: BCA
Internship organization: Glowlogics

CONTENTS
- SentinelGuard_Professional_Project_Report.docx : formatted project report
- sentinelguard_project.py : complete Python source
- banking_fraud_dataset.csv : generated synthetic dataset
- sentinelguard_model.pkl : trained model
- confusion_matrix.png : evaluation chart
- feature_importance.png : explainability chart
- sample_test_predictions.csv : sample held-out predictions

RUN IN GOOGLE COLAB
1. Open a new Colab notebook.
2. Run: !pip install -q numpy pandas matplotlib scikit-learn joblib
3. Upload sentinelguard_project.py to the Colab session.
4. Run: %run sentinelguard_project.py

IMPORTANT
This is an academic demonstration using synthetic data and generated labels.
It is not a real-time bank integration and is not suitable for actual financial
risk decisions. Please compare the report with Glowlogics' official project brief
and add any company-required items before final submission.

OBSERVED TEST METRICS ON THIS REPRODUCIBLE SYNTHETIC DATASET
- Accuracy: 0.7765
- Precision: 0.5867
- Recall: 0.2391
- F1-score: 0.3397
- ROC-AUC: 0.7692