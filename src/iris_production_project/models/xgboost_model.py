from xgboost import XGBClassifier

# def train(X_train, y_train):
#     model = model = XGBClassifier()
#     model.fit(X_train, y_train)
#     return model

def train(X, y, params=None):
    params = params or {}

    model = XGBClassifier(
        learning_rate=params.get("learning_rate", 0.1),
        max_depth=int(params.get("max_depth", 6)),
        n_estimators=int(params.get("n_estimators", 100)),
        subsample=params.get("subsample", 1.0),
        colsample_bytree=params.get("colsample_bytree", 1.0),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)
    return model