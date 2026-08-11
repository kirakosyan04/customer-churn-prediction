import joblib
import numpy as np

from src.preprocessing import load_data


def test_saved_pipeline_exists():
    pipeline = joblib.load("models/churn_pipeline.joblib")

    assert pipeline is not None
    assert "model" in pipeline.named_steps


def test_model_prediction_shape():
    X, _ = load_data()

    pipeline = joblib.load("models/churn_pipeline.joblib")

    sample = X.head(10)
    probabilities = pipeline.predict_proba(sample)[:, 1]

    assert len(probabilities) == 10
    assert np.all((probabilities >= 0) & (probabilities <= 1))