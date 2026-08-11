import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from preprocessing import load_data, create_preprocessor
from xgboost import XGBClassifier


def run_cross_validation():
    X, y = load_data()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        preprocessor = create_preprocessor(X_train)

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
            random_state=42,
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_val)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        fold_results.append({
            "fold": fold,
            "f1": f1_score(y_val, predictions),
            "roc_auc": roc_auc_score(y_val, probabilities),
        })

    results = pd.DataFrame(fold_results)

    print(results)
    print("\nMean:")
    print(results[["f1", "roc_auc"]].mean())

    print("\nStandard deviation:")
    print(results[["f1", "roc_auc"]].std())


if __name__ == "__main__":
    run_cross_validation()