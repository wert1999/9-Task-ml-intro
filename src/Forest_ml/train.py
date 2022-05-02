from pathlib import Path

import click
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@click.command()
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


# function to train a given model, generate predictions, and return accuracy score
def fit_evaluate_model(model, X_train, y_train, X_valid, Y_valid):
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_valid)
    return accuracy_score(Y_valid, y_predicted)


def train(dataset_path: Path, random_state: int, test_split_ratio: float) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
    # create scaler
    scaler = StandardScaler()
    # apply normalization to training set and transform training set
    features_train_scaled = scaler.fit_transform(features_train, target_train)
    # transform validation set
    features_valid_scaled = scaler.transform(features_val)
    
    knn_classifier = KNeighborsClassifier()
    knn_accuracy = fit_evaluate_model(knn_classifier, features_train_scaled, target_train, features_valid_scaled, target_val)
    
    click.echo(f"Accuracy: {knn_accuracy}.")
#train()