import pandas as pd

from src.feature_engineering import ChurnFeatureEngineer


def test_feature_engineering():
    data = pd.DataFrame({
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "OnlineSecurity": ["Yes"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["Yes"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["No"],
        "TotalCharges": [1000.0],
        "tenure": [10]
    })

    result = ChurnFeatureEngineer().transform(data)

    assert "ServiceCount" in result.columns
    assert "AvgMonthlyCharges" in result.columns
    assert result["ServiceCount"].iloc[0] == 4