from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import StandardScaler


def create_pipeline() -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)

    rfc = RandomForestClassifier()


rfc_param = {
    "max_depth": [3, None],
    "n_estimators": [5, 10, 50, 100],
    "max_features": ["auto", "sqrt", "log2"],
    "criterion": ["gini", "entropy"],
    "bootstrap": [True, False],
}
grid = GridSearchCV(
    estimator=rfc, param_grid=rfc_param, scoring="accuracy", cv=10, n_jobs=-1
)
start = time()
result = grid.fit(X_train.values, y_train.values)
# rfc.fit(X_train.values, y_train.values)
print("Best Score: %s" % result.best_score_)
print("Best Hyperparameters: %s" % result.best_params_)
y_pred = grid.predict(X_val.values)
print("accuracy on validation data=", accuracy_score(y_val, y_pred))
print("Time", time() - start)


max_features = (max_features,)
