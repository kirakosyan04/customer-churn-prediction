import pandas as pd
import shap


def explain_xgboost(pipeline, X):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names
    )