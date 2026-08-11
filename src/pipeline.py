import logging

import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from feature_engineering import ChurnFeatureEngineer
from preprocessing import create_preprocessor, load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def build_pipeline(X):
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
        ("feature_engineering", ChurnFeatureEngineer()),
        ("preprocessor", create_preprocessor(X)),
        ("model", model)
    ])


def train_and_save():
    logger.info("Loading dataset")
    X, y = load_data()

    logger.info("Building pipeline")
    pipeline = build_pipeline(X)

    logger.info("Training model")
    pipeline.fit(X, y)

    logger.info("Saving model")
    joblib.dump(
        pipeline,
        "models/churn_pipeline.joblib"
    )

    logger.info("Pipeline saved successfully")


if __name__ == "__main__":
    train_and_save()