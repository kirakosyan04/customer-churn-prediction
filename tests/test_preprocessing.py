from src.preprocessing import create_preprocessor, load_data


def test_load_data():
    X, y = load_data()

    assert len(X) == len(y)
    assert len(X) > 0
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns


def test_preprocessor():
    X, _ = load_data()

    preprocessor = create_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0