# 🏦 Loan Approval Prediction - Machine Learning Project

This is a complete end-to-end machine learning project that predicts whether a loan application will be approved based on applicant data. The project uses a Random Forest classifier and includes an interactive dashboard to explain model predictions using SHAP values.

---

## 📌 Project Objective

To automate loan application decision-making by:
- Predicting loan approval (classification)
- Explaining model behavior with SHAP
- Allowing real-time “What-if” analysis via dashboard

---

## 📊 Features

- Clean and preprocess real-world tabular data
- Train a RandomForestClassifier model
- Evaluate performance with metrics
- Visualize feature importances
- Interpret results using SHAP values
- Launch an interactive dashboard using ExplainerDashboard

---

## 🧰 Tech Stack & Libraries

- **Language:** Python 3.11
- **ML Framework:** scikit-learn
- **Visualization:** SHAP, matplotlib
- **Dashboarding:** ExplainerDashboard
- **Others:** pandas, numpy, joblib

---

## 📁 Project Structure

  loan_dataset.csv # Input dataset
  preprocess.py # Data loading and preprocessing
  trainmodel.py # Model training and evaluation
  dashboard.py # Launches interactive dashboard
  loan_model.pkl # Saved trained model
