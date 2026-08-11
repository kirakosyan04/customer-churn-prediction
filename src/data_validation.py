import pandas as pd


class DataValidator:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def check_missing_values(self):
        return self.data.isnull().sum()

    def check_duplicates(self):
        return self.data.duplicated().sum()

    def check_data_types(self):
        return self.data.dtypes

    def check_target_distribution(self, target_col: str):
        return self.data[target_col].value_counts(normalize=True)

    def validate(self, target_col: str):
        return {
            "missing_values": self.check_missing_values(),
            "duplicates": self.check_duplicates(),
            "data_types": self.check_data_types(),
            "target_distribution": self.check_target_distribution(target_col),
        }