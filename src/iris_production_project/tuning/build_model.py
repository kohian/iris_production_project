from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_model(args):
    if args.model == "logreg":
        return LogisticRegression(
            C=args.C,
            solver="lbfgs",
            max_iter=1000,
        )

    elif args.model == "xgb":
        return XGBClassifier(
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            n_estimators=args.n_estimators,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")