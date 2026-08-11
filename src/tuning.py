import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import load_data, create_preprocessor


def tune_xgboost():
    X, y = load_data()
    preprocessor = create_preprocessor(X)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ))
    ])

    param_grid = {
        "model__n_estimators": [200, 300, 500, 700],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=20,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X, y)

    print("Best F1:", search.best_score_)
    print("Best parameters:")
    print(search.best_params_)


if __name__ == "__main__":
    tune_xgboost()