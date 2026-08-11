import pandas as pd


class ChurnFeatureEngineer:
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for churn prediction."""
        df = data.copy()

        # Count the number of subscribed services
        service_columns = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        df["ServiceCount"] = (
          df[service_columns]
          .apply(
              lambda col: col.map({
                 "Yes": 1,
                 "No": 0,
                 "No phone service": 0,
                 "No internet service": 0
              })
          )
          .fillna(0)
          .sum(axis=1)
)
        # Calculate the average monthly charge over the customer's tenure
        df["AvgMonthlyCharges"] = (
            df["TotalCharges"] / df["tenure"].replace(0, 1)
        )

        return df