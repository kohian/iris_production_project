import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # model selection
    parser.add_argument("--model", choices=["logreg", "xgb"], default="logreg")

    # Logistic Regression param
    parser.add_argument("--C", type=float, default=1.0)

    # XGBoost params
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)

    return parser.parse_args()