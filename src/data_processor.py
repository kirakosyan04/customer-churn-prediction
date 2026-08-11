import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ChurnDataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self) -> pd.DataFrame:
        """Load the data and perform initial cleaning."""
        self.data = pd.read_csv(self.file_path)

        # Convert empty strings to missing values
        self.data = self.data.replace(r'^\s*$', np.nan, regex=True)

        # Convert TotalCharges to numeric values
        self.data["TotalCharges"] = pd.to_numeric(
            self.data["TotalCharges"], errors="coerce"
        )

        # Remove rows with missing values
        self.data = self.data.dropna()

        return self.data

    def preprocess(self) -> pd.DataFrame:
        """Encode categorical variables using Label Encoding."""
        if self.data is None:
            raise ValueError("Data has not been loaded. Call load_data() first.")

        processed_data = self.data.copy()

        # Remove irrelevant columns, such as customer IDs
        if "customerID" in processed_data.columns:
            processed_data = processed_data.drop("customerID", axis=1)

        # Encode categorical columns as numerical values
        for column in processed_data.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            processed_data[column] = le.fit_transform(
                processed_data[column].astype(str)
            )
            self.label_encoders[column] = le

        self.data = processed_data
        return self.data

    def get_splits(self, target_col: str, test_size: float = 0.2):
        """Split the data into training and test sets and standardize the features."""
        if self.data is None:
            raise ValueError(
                "Data has not been preprocessed. Call preprocess() first."
            )

        X = self.data.drop(target_col, axis=1)
        y = self.data[target_col]

        # Standardize the features for model training
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        return train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )