from sklearn.linear_model import LogisticRegression

# def train(X_train, y_train):
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)
#     return model


def train(X, y, params=None):
    params = params or {}

    model = LogisticRegression(
        C=params.get("C", 1.0),
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(X, y)
    return model