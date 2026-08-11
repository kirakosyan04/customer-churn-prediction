import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import create_preprocessor, load_data


def compare_models():
    X, y = load_data()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=5,
            eval_metric="logloss",
            random_state=42
        ),
    }

    results = []

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", create_preprocessor(X)),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=["f1", "roc_auc", "precision", "recall"],
            n_jobs=-1
        )

        results.append({
            "Model": name,
            "F1": scores["test_f1"].mean(),
            "ROC-AUC": scores["test_roc_auc"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
        })

    results_df = pd.DataFrame(results)
    print(results_df.round(4))


if __name__ == "__main__":
    compare_models()