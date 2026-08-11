import pandas as pd

from src.data_validation import DataValidator


def test_data_validation():
    data = pd.DataFrame({
        "tenure": [1, 12, 24],
        "MonthlyCharges": [29.85, 56.95, 70.35],
        "Churn": ["Yes", "No", "No"]
    })

    validator = DataValidator(data)

    assert validator.check_duplicates() == 0
    assert validator.check_missing_values().sum() == 0
    assert validator.check_target_distribution("Churn").sum() == 1