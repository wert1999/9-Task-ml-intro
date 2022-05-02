from pathlib import Path
from joblib import dump


import click
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# function to train a given model, generate predictions, and return accuracy score
def fit_evaluate_model(model, X_train, y_train, X_valid, Y_valid):
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_valid)
    return accuracy_score(Y_valid, y_predicted)


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
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)

def train(dataset_path: Path, save_model_path: Path, random_state: int, test_split_ratio: float) -> None:
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
    
    knn = KNeighborsClassifier()
    knn.fit(features_train_scaled, target_train)
    y_predicted_knn = knn.predict(features_valid_scaled)
    knn_accuracy = accuracy_score(target_val, y_predicted_knn)
    click.echo(f"Accuracy KNN model: {knn_accuracy}.")

    knn_path = save_model_path
    knn_path = Path(knn_path.parent, f"{knn_path.stem}_knn{knn_path.suffix}")
    print("save_model_path",save_model_path)

    print("knn_path",knn_path)
    dump(knn, knn_path)

    click.echo(f"Model is saved to {knn_path}.")

    rf = RandomForestClassifier()
    rf.fit(features_train, target_train)
    y_predicted_rf = rf.predict(features_val)
    rf_accuracy = accuracy_score(target_val, y_predicted_rf)
    click.echo(f"Accuracy RandomForest model: {rf_accuracy}.")
    rf_path = save_model_path
    rf_path = Path(rf_path.parent, f"{rf_path.stem}_rf{rf_path.suffix}")
    dump(rf, rf_path)
    click.echo(f"Model is saved to {rf_path}.")
