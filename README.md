[![CI](https://github.com/kirakosyan04/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/kirakosyan04/customer-churn-prediction/actions/workflows/ci.yml)
# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using the Telco Customer Churn dataset.

## Overview

The project covers the complete machine learning workflow:

- Data validation
- Feature engineering
- Preprocessing
- Cross-validation
- Model comparison
- Hyperparameter tuning
- Threshold optimization
- Probability calibration
- Error analysis
- SHAP explainability
- Automated testing
- Reproducible model configuration
- Final model pipeline

## Dataset

The project uses the Telco Customer Churn dataset.

The target variable is `Churn`, which indicates whether a customer leaves the service.

The raw dataset is excluded from Git using `.gitignore`.

## Project Structure

customer-churn-prediction/
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_pipeline.joblib
│
├── notebooks/
│
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
│
├── tests/
│   ├── test_data_validation.py
│   ├── test_feature_engineering.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── requirements.txt
└── README.md


## Machine Learning Workflow

Raw Data
   ↓
Data Validation
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
Cross-Validation
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Threshold Optimization
   ↓
Calibration
   ↓
Final Evaluation
   ↓
Explainability
   ↓
Model Saving

## Feature Engineering

Two additional features are created:

### ServiceCount

Counts the number of subscribed services for each customer.

### AvgMonthlyCharges

Calculates the average monthly charge based on `TotalCharges` and `tenure`.

## Models

Three classification models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

| Model | F1 | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5996 | 0.8451 | 0.6575 | 0.5511 |
| Random Forest | 0.6357 | 0.8448 | 0.5630 | 0.7303 |
| XGBoost | 0.5929 | 0.8475 | 0.6667 | 0.5340 |

## Cross-Validation

XGBoost was evaluated using 5-fold stratified cross-validation.

| Metric | Mean | Std |
|---|---:|---:|
| F1 | 0.5946 | 0.0144 |
| ROC-AUC | 0.8468 | 0.0048 |

## Hyperparameter Tuning

XGBoost hyperparameters were tuned using cross-validation.

Best parameters:

- n_estimators = 300
- max_depth = 3
- learning_rate = 0.05
- subsample = 0.9
- colsample_bytree = 0.8
- min_child_weight = 5

Best cross-validation F1: 0.5929

## Threshold Optimization

The classification threshold was optimized for F1 instead of using the default 0.5.

Best threshold: 0.39

| Metric | Score |
|---|---:|
| F1 | 0.6247 |
| Precision | 0.5803 |
| Recall | 0.6765 |

## Final Evaluation

Using the optimized threshold of 0.39:

| Metric | Score |
|---|---:|
| Accuracy | 0.7839 |
| Precision | 0.5803 |
| Recall | 0.6765 |
| F1 | 0.6247 |
| ROC-AUC | 0.8376 |
| Brier Score | 0.1401 |

## Error Analysis

At the optimized threshold:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 850 | 183 |
| Actual Yes | 121 | 253 |

- True Negatives: 850
- False Positives: 183
- False Negatives: 121
- True Positives: 253

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

Train and save the final pipeline:

PYTHONPATH=src python3 src/pipeline.py

The trained pipeline is saved to:

models/churn_pipeline.joblib

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
