import joblib

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import load_data, create_preprocessor
from feature_engineering import ChurnFeatureEngineer


def build_pipeline(X):
    feature_engineer = ChurnFeatureEngineer()

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

    return Pipeline([
        ("feature_engineering", feature_engineer),
        ("preprocessor", create_preprocessor(X)),
        ("model", model)
    ])


def train_and_save():
    X, y = load_data()

    pipeline = build_pipeline(X)
    pipeline.fit(X, y)

    joblib.dump(
        pipeline,
        "models/churn_pipeline.joblib"
    )

    print("Pipeline saved to models/churn_pipeline.joblib")


if __name__ == "__main__":
    train_and_save()