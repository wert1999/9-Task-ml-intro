from pathlib import Path

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
    classifier = LogisticRegression(random_state=random_state).fit(
        features_train, target_train
    )
    accuracy = accuracy_score(target_val, classifier.predict(features_val))
    click.echo(f"Accuracy: {accuracy}.")
train()