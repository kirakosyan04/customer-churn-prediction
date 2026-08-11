import shap

def explain_xgboost(model, X_test, preprocessor):
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names
    )
