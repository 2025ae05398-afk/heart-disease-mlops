# End-to-End MLOps for Heart Disease Classification

**Course:** AIMLCZG523 | **Student:** [KULDEEP SHARMA/2025AE05398] | **Date:** [19/10/2026] | **Repository:** [https://github.com/2025ae05398-afk/heart-disease-mlops]

> Educational software demonstration only. Not a medical diagnosis.

## 1. Executive summary
State the objective, exact dataset version, selected model and actual test metrics from `artifacts/metrics.json`.

## 2. Problem statement
Describe binary classification as an MLOps lifecycle exercise, not a clinical product.

## 3. Dataset and acquisition
Record UCI dataset ID 45, retrieval date, row/column count, target column and mapping, missing values and feature types.

## 4. Exploratory data analysis
Insert `artifacts/eda/class_balance.png`, `feature_histograms.png`, and `correlation_heatmap.png`. Discuss patterns actually seen in your run.

## 5. Feature engineering and modelling
Explain numeric median imputation/scaling, categorical mode imputation/one-hot encoding, Logistic Regression and Random Forest, and five-fold stratified GridSearchCV.

## 6. Evaluation
Insert actual accuracy, precision, recall, ROC-AUC and CV ROC-AUC from `artifacts/metrics.json`; include confusion matrix and ROC curve. State positive class and do not claim clinical validity.

## 7. Experiment tracking
Include MLflow screenshots and explain parameters, CV metrics, held-out metrics, model artifacts and plots logged.

## 8. Packaging and reproducibility
Explain the saved complete pipeline, requirements, seeds, dataset script and clean setup instructions.

## 9. CI/CD and testing
Include workflow URL/screenshots and actual Ruff/pytest results. Explain that failed checks stop the workflow.

## 10. API, Docker, deployment and monitoring
Show `/health`, `/predict`, `/metrics`; Docker build/run proof; Kubernetes pods/service or cloud deployment proof if used. Do not describe local testing as public cloud deployment.

## 11. Limitations, security and ethics
The dataset is small and may not represent current populations. High benchmark performance does not establish clinical utility. Real services need authentication, HTTPS, privacy controls and clinical validation.

## 12. Conclusion
Summarize actual results, workflow and future work.

## Submission evidence checklist
- [ ] EDA plots and summary
- [ ] MLflow experiment screenshot
- [ ] pytest and Ruff output
- [ ] GitHub Actions successful run
- [ ] Docker build and health endpoint
- [ ] Successful prediction request/response
- [ ] Prometheus metrics output
- [ ] Kubernetes/cloud deployment proof if claimed
- [ ] Repository URL and short pipeline demo video
