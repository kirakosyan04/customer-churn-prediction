from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import create_preprocessor, load_data


def evaluate_calibration():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    base_model = Pipeline([
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

    calibrated_model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=5
    )

    calibrated_model.fit(X_train, y_train)

    probabilities = calibrated_model.predict_proba(X_test)[:, 1]

    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    print("Brier Score:", round(brier_score_loss(y_test, probabilities), 4))


if __name__ == "__main__":
    evaluate_calibration()