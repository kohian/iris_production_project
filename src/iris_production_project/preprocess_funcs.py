# from pathlib import Path
# from src.config import RAW_PATH
import pandas as pd

from iris_production_project.config import PROCESSED_PATH, RAW_PATH

# # RAW_PATH = Path("data/raw/iris.csv")
# PROCESSED_PATH = Path("data/processed/iris_processed.csv")
# RAW_PATH = "gs://iris-csv/data/iris.csv"
# PROCESSED_PATH = "gs://iris-csv/data/processed/iris_processed.csv"

def load_raw_data():
    """Load the raw dataset."""
    df = pd.read_csv(RAW_PATH)
    return df

def clean_data(df):
    """Basic cleaning / validation."""
    
    # remove duplicate rows
    duplicates = df[df.duplicated()]
    if duplicates.empty:
        print("No duplicates found")
    else:
        print("Duplicates found:")
        print(duplicates)
    df = df.drop_duplicates()

    # standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # check for missing values
    missing = df.isna().sum()

    if missing.sum() > 0:
        raise ValueError(f"Missing values detected:\n{missing[missing > 0]}")

    print("Dataset processed successfully")
    return df


def save_processed_data(df):
    """Save processed dataset."""    
    # PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved to: {PROCESSED_PATH}")


def load_processed_data():
    """Load the dataset."""
    df = pd.read_csv(PROCESSED_PATH)
    return df


def split_features_target(df):
    X = df.iloc[:, :-2]
    y = df.iloc[:, -2]
    return X, y


def get_named_target(df):
    y_named = df.iloc[:, -1]
    return y_named


