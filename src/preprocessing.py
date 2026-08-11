import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data():
    """Load and prepare the raw dataset."""
    df = pd.read_csv(
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    df = df.dropna()

    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    df = df.drop(columns=["customerID"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def create_preprocessor(X):
    """Create a preprocessing pipeline for numerical and categorical features."""
    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                categorical_features,
            ),
        ]
    )