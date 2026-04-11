import argparse
import json

import gcsfs
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate

from iris_production_project.config import MODEL_DIR
from iris_production_project.log_metrics import log_metrics
from iris_production_project.models import logistic_regression_model, xgboost_model
from iris_production_project.preprocess_funcs import load_processed_data, split_features_target

MODEL_REGISTRY = {
    "logreg": logistic_regression_model.train,
    "xgb": xgboost_model.train,
}


GCS_BASE_PATH = "gs://iris-csv/tuning"


def load_best_params(model: str, params_file: str | None) -> dict:
    if not params_file:
        return {}

    full_path = f"{GCS_BASE_PATH}/{model}/{params_file}"
    print(f"Loading tuned params from: {full_path}")

    fs = gcsfs.GCSFileSystem()
    with fs.open(full_path, "r") as f:
        payload = json.load(f)

    best_params = payload.get("best_params", {})
    print(f"Best params: {best_params}")
    return best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="logreg")
    parser.add_argument("--params-file", type=str, default=None)
    args = parser.parse_args()

    if args.params_file:
        run_name = args.params_file.replace(".json", "")
    else:
        run_name = "baseline"

    # load tuned params if provided
    best_params = load_best_params(args.model, args.params_file)

    # load data
    df = load_processed_data()
    X, y = split_features_target(df)

    # train final model
    train_fn = MODEL_REGISTRY[args.model]
    model = train_fn(X, y, params=best_params)


    # save final model

    model_path = f"{MODEL_DIR}/{args.model}/{run_name}.joblib"
    fs = gcsfs.GCSFileSystem()
    with fs.open(model_path, "wb") as f:
        joblib.dump(model, f)

    print(f"Saved model to: {model_path}")

    # evaluate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    scores = cross_validate(model, X, y, cv=skf, scoring=metrics)

    log_metrics(scores, args.model, run_name)

    print("Metrics logged to reports folder")


if __name__ == "__main__":
    main()

# python -m iris_production_project.train_evaluate --model xgb
# python -m iris_production_project.train_evaluate --model xgb --params-file 20260410-151741.json