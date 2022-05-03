from pathlib import Path
from joblib import dump

from numpy import mean
import mlflow
import mlflow.sklearn
import click
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


'''
# function to train a given model, generate predictions, and return accuracy score
def fit_evaluate_model(model, X_train, y_train, X_valid, Y_valid):
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_valid)
    return accuracy_score(Y_valid, y_predicted)
'''

@click.command()
@click.option(
    "-e",
    "--estimator",
    default="rf",
    type=click.Choice(['rf', 'knn']),
)
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--random-state", default=42, type=int)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "-n",
    "--n-estimators",
    default=100,
    help='n_estimators for RF / n_neighbors for KNN ',
    type=int,
    show_default=True,
)
@click.option(
    "-c",
    "--criterion",
    default='gini',
    type=click.Choice(['gini', 'entropy']),
    show_default=True,
)
@click.option(
    "-m",
    "--max-depth",
    default=None,
    type= int,
    show_default=True,
)
@click.option(
    "-cv",
    "--cv",
    default=5,
    type= int,
    show_default=True,
)
@click.option(
    "-sc",
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "-w",
    "--weights",
    default='uniform',
    type=click.Choice(['uniform', 'distance']),
    show_default=True,
)

def train(estimator, dataset_path: Path, save_model_path: Path, random_state: int, test_split_ratio: float,
    n_estimators: int, criterion, max_depth: int, cv:int, use_scaler: bool, weights) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio)

    with mlflow.start_run():
        if estimator == "knn":
            knn = KNeighborsClassifier(n_neighbors=n_estimators, weights=weights)

            # create scaler
            if use_scaler:
                scaler = StandardScaler()
                features_train_scaled = scaler.fit_transform(features_train, target_train)
                features_valid_scaled = scaler.transform(features_val)
                knn.fit(features_train_scaled, target_train)
                y_predicted_knn = knn.predict(features_valid_scaled)
            else:
                knn.fit(features_train, target_train)
                y_predicted_knn = knn.predict(features_val)    

            knn_accuracy = accuracy_score(target_val, y_predicted_knn)
            click.echo(f"Accuracy KNN model w/o CV: {knn_accuracy}.")

            scores_accuracy = cross_val_score(knn,features, target, scoring='accuracy', cv=cv, n_jobs=-1)
            scores_v_measure = cross_val_score(knn,features, target, scoring='v_measure_score', cv=cv, n_jobs=-1)
            scores_f1_micro = cross_val_score(knn,features, target, scoring='f1_micro', cv=cv, n_jobs=-1)

            mlflow.log_param("model", estimator)
            mlflow.log_param("n_neighbors", n_estimators)
            mlflow.log_param("use_scaler", use_scaler)
            mlflow.log_param("weights", weights)
            mlflow.log_metric("acc. w.o CV", knn_accuracy)
            mlflow.log_metric("accuracy", mean(scores_accuracy))
            mlflow.log_metric("v_measure", mean(scores_v_measure))
            mlflow.log_metric("f1_micro", mean(scores_f1_micro))

            knn_path = save_model_path
            knn_path = Path(knn_path.parent, f"{knn_path.stem}_knn{knn_path.suffix}")
            print("save_model_path",save_model_path)
            print("knn_path",knn_path)
            dump(knn, knn_path)
            click.echo(f"Model is saved to {knn_path}.")

        elif estimator == "rf":

            rf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                max_depth=max_depth, random_state=random_state) #,max_features=max_features

            rf.fit(features_train, target_train)
            y_predicted_rf = rf.predict(features_val)
            rf_accuracy = accuracy_score(target_val, y_predicted_rf)

            scores_accuracy = cross_val_score(rf,features, target, scoring='accuracy', cv=cv, n_jobs=-1)
            scores_v_measure = cross_val_score(rf,features, target, scoring='v_measure_score', cv=cv, n_jobs=-1)
            scores_f1_micro = cross_val_score(rf,features, target, scoring='f1_micro', cv=cv, n_jobs=-1)
            click.echo(f"Accuracy RandomForest model w/o CV: {rf_accuracy}.")

            mlflow.log_param("model", estimator)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("acc. w.o CV", rf_accuracy)
            mlflow.log_metric("accuracy", mean(scores_accuracy))
            mlflow.log_metric("v_measure", mean(scores_v_measure))
            mlflow.log_metric("f1_micro", mean(scores_f1_micro))

            rf_path = save_model_path
            rf_path = Path(rf_path.parent, f"{rf_path.stem}_rf{rf_path.suffix}")
            dump(rf, rf_path)
            click.echo(f"Model is saved to {rf_path}.")



'''
@click.option(
    "-f",
    "--max-features",
    default="auto",
    type=click.Choice(['auto', 'sqrt', 'log2', click.INT, click.FLOAT], case_sensitive=False),
    show_default=True,
)
'''