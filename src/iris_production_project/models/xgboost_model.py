from xgboost import XGBClassifier


def train(X_train, y_train):

    model = model = XGBClassifier()
    model.fit(X_train, y_train)

    return model