import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import load_data, create_preprocessor
from feature_engineering import ChurnFeatureEngineer


def evaluate_model(X, y, tuned=False):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    if tuned:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=5,
            eval_metric="logloss",
            random_state=42
        )
    else:
        model = XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42
        )

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor(X)),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=["f1", "roc_auc"],
        n_jobs=-1
    )

    return {
        "F1": scores["test_f1"].mean(),
        "ROC-AUC": scores["test_roc_auc"].mean()
    }


def run_ablation():
    X, y = load_data()

    results = []

    baseline = evaluate_model(X, y)

    results.append({
        "Experiment": "Baseline",
        **baseline
    })

    X_engineered = ChurnFeatureEngineer().transform(X.copy())
    engineered = evaluate_model(X_engineered, y)

    results.append({
        "Experiment": "Feature Engineering",
        **engineered
    })

    tuned = evaluate_model(X, y, tuned=True)

    results.append({
        "Experiment": "Tuned XGBoost",
        **tuned
    })

    results_df = pd.DataFrame(results)

    print(results_df.round(4))


if __name__ == "__main__":
    run_ablation()