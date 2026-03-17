# from pathlib import Path
from src.config import MODEL_DIR
import gcsfs
import argparse
import joblib
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from src.preprocess_funcs import load_processed_data, split_features_target, get_named_target
from src.models import logistic_regression_model, xgboost_model
from src.log_metrics import log_metrics

# MODEL_DIR = Path("model_artifacts")

MODEL_REGISTRY = {
    "logreg": logistic_regression_model.train,
    "xgb": xgboost_model.train,
}

def main():
    # specify model
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="logreg")
    args = parser.parse_args()

    # load data
    df = load_processed_data()
    X, y = split_features_target(df)
    
    # Train and save full model
    train_fn = MODEL_REGISTRY[args.model]
    model = train_fn(X, y)

    # MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = f"{MODEL_DIR}/{args.model}.joblib"
    # joblib.dump(model, model_path)
    fs = gcsfs.GCSFileSystem()
    with fs.open(model_path, "wb") as f:
        joblib.dump(model, f)

    print(f"Saved model to: {model_path}")

    # Evaluate model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    scores = cross_validate(model, X, y, cv=skf, scoring=metrics)

    log_metrics(scores,args.model)
    print(f"Metrics logged to reports folder")

if __name__ == "__main__":
    main()