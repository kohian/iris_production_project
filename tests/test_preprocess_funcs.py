import pandas as pd
import pytest

from iris_production_project.preprocess_funcs import (
    clean_data,
    get_named_target,
    split_features_target,
)


def test_clean_data_standardizes_column_names_and_drops_duplicates():
    df = pd.DataFrame(
        {
            "Sepal Length": [5.1, 5.1, 4.9],
            "Petal Width ": [0.2, 0.2, 0.2],
            "Target": [0, 0, 0],
            "Species Name": ["setosa", "setosa", "setosa"],
        }
    )

    result = clean_data(df)

    assert list(result.columns) == [
        "sepal_length",
        "petal_width",
        "target",
        "species_name",
    ]
    assert len(result) == 2


def test_clean_data_raises_when_missing_values_exist():
    df = pd.DataFrame(
        {
            "A": [1.0, None],
            "B": [2.0, 3.0],
            "target": [0, 1],
            "species": ["x", "y"],
        }
    )

    with pytest.raises(ValueError, match="Missing values detected"):
        clean_data(df)


def test_split_features_target_uses_second_last_column_as_target():
    df = pd.DataFrame(
        {
            "f1": [1, 2],
            "f2": [3, 4],
            "target": [0, 1],
            "species_name": ["setosa", "versicolor"],
        }
    )

    X, y = split_features_target(df)

    assert list(X.columns) == ["f1", "f2"]
    assert y.tolist() == [0, 1]


def test_get_named_target_returns_last_column():
    df = pd.DataFrame(
        {
            "f1": [1, 2],
            "f2": [3, 4],
            "target": [0, 1],
            "species_name": ["setosa", "versicolor"],
        }
    )

    y_named = get_named_target(df)

    assert y_named.tolist() == ["setosa", "versicolor"]