[![CI](https://github.com/kirakosyan04/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/kirakosyan04/customer-churn-prediction/actions/workflows/ci.yml)
# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using the Telco Customer Churn dataset.

## Overview

The project covers the complete machine learning workflow:

- Data validation
- Feature engineering
- Preprocessing
- Model comparison
- Threshold optimization
- Probability calibration
- SHAP explainability
- Automated testing
- Reproducible model configuration
- Final model pipeline

## Dataset

This project uses the **Telco Customer Churn** dataset





- Samples: 7,043
- Features: 20 input features
- Target: Churn
- Task: Binary classification

## Project Structure

```text
customer-churn-prediction/
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── churn_pipeline.joblib
│   └── random_forest_model.json
├── src/
│   ├── ablation_study.py
│   ├── calibration.py
│   ├── config.py
│   ├── cross_validation.py
│   ├── data_processor.py
│   ├── data_validation.py
│   ├── error_analysis.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── feature_engineering.py
│   ├── final_evaluation.py
│   ├── model_comparison.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── threshold_optimization.py
│   ├── train.py
│   └── tuning.py
├── tests/
│   ├── test_data_validation.py
│   ├── test_feature_engineering.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Feature Engineering

Two additional features are created:

### ServiceCount

Counts the number of subscribed services for each customer.

### AvgMonthlyCharges

Calculates the average monthly charge based on `TotalCharges` and `tenure`.

## Models

Three classification models were evaluated using stratified 5-fold cross-validation:

- Logistic Regression
- Random Forest
- XGBoost

| Model | F1 | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5996 | 0.8451 | 0.6575 | 0.5511 |
| **Random Forest** | **0.6357** | 0.8448 | 0.5630 | **0.7303** |
| XGBoost | 0.5929 | **0.8475** | **0.6667** | 0.5340 |

Random Forest achieved the highest F1-score and recall among the evaluated models.

## Final Model

Random Forest was selected as the final model based on its performance in the model comparison.

The main objective is to identify customers who are likely to churn, making recall particularly important. A false negative means that a customer who is likely to churn was not identified.

Random Forest achieved the highest recall and F1-score among the evaluated models:

- **Recall:** 0.7380
- **F1-score:** 0.6287
- **Precision:** 0.5476
- **ROC-AUC:** 0.8358
- **Accuracy:** 0.7683

Therefore, Random Forest was selected as the final model.

## Threshold Optimization for XGBoost

The classification threshold was optimized for XGBoost to improve churn detection performance.

Instead of using the default threshold of 0.5, lower thresholds were evaluated.

| Threshold | F1 | Precision | Recall |
|---:|---:|---:|---:|
| 0.50 | 0.5776 | 0.6317 | 0.5321 |
| 0.40 | 0.6253 | 0.5833 | 0.6738 |
| **0.38** | **0.6283** | 0.5696 | **0.7005** |

Threshold optimization improved XGBoost's recall and F1-score. However, Random Forest still achieved higher recall (0.7380) and slightly higher F1-score (0.6287), so Random Forest was selected as the final model.

## Model Explainability

SHAP is used to analyze the contribution of features to XGBoost predictions.

This provides both global feature importance and insight into how individual features influence churn predictions.

## Ablation Study

Feature engineering was evaluated against the baseline model.

| Experiment | F1 | ROC-AUC |
|---|---:|---:|
| Baseline | 0.5929 | 0.8475 |
| Feature Engineering | 0.5880 | 0.8471 |

The engineered features did not improve the model in this experiment.

## Testing

The project contains automated tests for:

- Data validation
- Feature engineering
- Preprocessing
- Saved model pipeline

Run all tests:

PYTHONPATH=src python3 -m pytest

Current test suite: 6 passed

## Training

Train the models using:

PYTHONPATH=src python3 src/train.py

The final Random Forest model is saved to:

models/random_forest_model.joblib

## Configuration

Experiment settings are centralized in:

src/config.py

The project uses fixed random seeds and stratified cross-validation for reproducibility.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- matplotlib
- pytest
- joblib
