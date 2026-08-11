# Customer Churn Prediction

Machine Learning project for predicting customer churn using Logistic Regression, Random Forest, and XGBoost, with model evaluation, threshold optimization, and SHAP explainability.

## Project Overview

The goal of this project is to predict whether a telecom customer is likely to churn.

The project includes:

* Data preprocessing and feature engineering
* Logistic Regression baseline
* Random Forest
* Tuned XGBoost
* Classification threshold optimization
* ROC-AUC evaluation
* SHAP-based model explainability
* Saved XGBoost model

## Project Structure

```text
customer-churn-prediction/
├── data/
│   └── raw/
├── models/
│   └── xgboost_model.json
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── .gitignore
└── README.md
```

## Models

Three classification models were evaluated:

* Logistic Regression
* Random Forest
* XGBoost

XGBoost achieved the highest ROC-AUC and was selected as the final model.

## Results

| Model               | Accuracy | Precision | Recall |    F1 |   ROC-AUC |
| ------------------- | -------: | --------: | -----: | ----: | --------: |
| Logistic Regression |    0.807 |     0.660 |  0.561 | 0.607 |     0.842 |
| Random Forest       |    0.763 |     0.540 |  0.727 | 0.620 |     0.840 |
| XGBoost             |    0.801 |     0.662 |  0.513 | 0.578 | **0.845** |

### Optimized Threshold

The default classification threshold was optimized to improve the F1-score.

* **Best threshold:** 0.33
* **Best F1-score:** 0.624

At threshold 0.4:

* Precision: 0.593
* Recall: 0.658
* F1: 0.624

## Explainability

SHAP is used to explain the XGBoost model and identify the features that have the greatest influence on churn predictions.

The most influential features include:

* Tenure
* Contract type
* Fiber optic internet service
* Monthly charges
* Electronic check payment method

The SHAP analysis provides insight into which customer characteristics increase or decrease predicted churn probability.

## Installation

Clone the repository:

```bash
git clone https://github.com/kirakosyan04/customer-churn-prediction.git
cd customer-churn-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the training pipeline:

```bash
python3 src/train.py
```

The pipeline:

1. Loads and preprocesses the dataset
2. Trains Logistic Regression, Random Forest, and XGBoost
3. Evaluates model performance
4. Optimizes the classification threshold
5. Generates the ROC curve
6. Generates the SHAP summary plot
7. Saves the trained XGBoost model

## Model

The trained XGBoost model is saved to:

```text
models/xgboost_model.json
```

## Dataset

This project uses the Telco Customer Churn dataset.

The raw dataset is excluded from Git tracking using `.gitignore`.

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* XGBoost
* SHAP
* Matplotlib
