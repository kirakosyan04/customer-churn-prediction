
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from evaluate import evaluate_model, evaluate_threshold, find_best_threshold
from preprocessing import create_preprocessor, load_data

# Load data
X, y = load_data()

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Preprocessing
preprocessor = create_preprocessor(X_train)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# ============================================================
# Logistic Regression
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]



print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# ============================================================
# Random Forest
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

joblib.dump(
    rf_model,
    "models/random_forest_model.joblib"
)
print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1:", f1_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))


# ============================================================
# Tuned XGBoost
# ============================================================

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42,
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\nTuned XGBoost")
evaluate_model(y_test, xgb_pred, xgb_prob)


# ============================================================
# XGBoost threshold optimization
# ============================================================

print("\nXGBoost - Threshold 0.4")
evaluate_threshold(
    y_test,
    xgb_prob,
    threshold=0.4,
)

best_threshold = find_best_threshold(
    y_test,
    xgb_prob,
)

print("\nBest threshold:", best_threshold)

best_pred = (xgb_prob >= best_threshold).astype(int)

print("\nXGBoost - Best Threshold")
print("Precision:", precision_score(y_test, best_pred))
print("Recall:", recall_score(y_test, best_pred))
print("F1:", f1_score(y_test, best_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_prob))


