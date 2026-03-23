import pandas as pd

from src.models.logistic_regression_model import train as train_logreg
from src.models.xgboost_model import train as train_xgb


def make_small_dataset():
    X = pd.DataFrame(
        {
            "f1": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
            "f2": [0.1, 0.2, 1.0, 1.1, 2.0, 2.1],
        }
    )
    y = [0, 0, 1, 1, 2, 2]
    return X, y


def test_logistic_regression_train_returns_fitted_model():
    X, y = make_small_dataset()
    model = train_logreg(X, y)

    assert hasattr(model, "predict")
    assert hasattr(model, "classes_")
    assert len(model.classes_) == 3


def test_xgboost_train_returns_fitted_model():
    X, y = make_small_dataset()
    model = train_xgb(X, y)

    assert hasattr(model, "predict")
    assert hasattr(model, "classes_")
    assert len(model.classes_) == 3