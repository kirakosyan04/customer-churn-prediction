import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import create_preprocessor, load_data


def optimize_threshold():
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor(X_train)),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=5,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_val)[:, 1]

    results = []

    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = (probabilities >= threshold).astype(int)

        results.append({
            "threshold": threshold,
            "f1": f1_score(y_val, predictions),
            "precision": precision_score(y_val, predictions),
            "recall": recall_score(y_val, predictions)
        })

    results_df = pd.DataFrame(results)

    best = results_df.loc[results_df["f1"].idxmax()]

    print("Best threshold:", round(best["threshold"], 2))
    print("F1:", round(best["f1"], 4))
    print("Precision:", round(best["precision"], 4))
    print("Recall:", round(best["recall"], 4))


if __name__ == "__main__":
    optimize_threshold()