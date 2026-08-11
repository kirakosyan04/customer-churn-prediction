RANDOM_STATE = 42

TEST_SIZE = 0.2

N_SPLITS = 5

THRESHOLD = 0.39

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}