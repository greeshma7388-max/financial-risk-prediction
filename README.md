# Financial Lifestyle Risk Score Prediction

Supervised machine learning pipeline to predict financial risk levels 
(Low / Medium / High) from behavioural financial indicators.

## Overview
Most personal finance tools are retrospective. This project builds a 
predictive model that classifies individuals into financial risk categories 
before adverse outcomes occur, using a novel composite risk score derived 
from financial behaviour ratios.

## Datasets
- Gen Z Money Spending Dataset — 1,700 records, 15 variables (Kaggle)
- Indian Personal Finance Dataset — 20,000 records, 27 variables (Kaggle)

## Key Results
| Model | Gen Z Accuracy | Indian Finance Accuracy |
|---|---|---|
| Logistic Regression | 87.35% | 97.58% |
| XGBoost (Tuned) | 81.18% | 96.15% |
| Random Forest (Tuned) | 79.71% | 94.00% |

- Best ROC-AUC: 0.9986 (Logistic Regression, Indian Finance dataset)
- Logistic Regression outperformed ensemble methods because the 
  composite risk score is a linear combination of features

## Techniques Used
- Feature engineering: composite Financial Lifestyle Risk Score from 
  savings rate, discretionary spend ratio, disposable income ratio, EMI burden
- Hyperparameter tuning: GridSearchCV with 5-fold stratified cross-validation
- What-if analysis: single-dimension improvements insufficient for 
  severely high-risk individuals

## Tools & Libraries
Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn

## Project Structure
financial_risk_prediction.py   # Gen Z pipeline
indian_finance_analysis.py     # Indian Finance pipeline
generate_missing_plots.py      # Plot generation
