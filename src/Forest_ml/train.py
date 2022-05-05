from pathlib import Path
from secrets import choice
from typing import Any
from joblib import dump
from numpy import mean
import mlflow
import mlflow.sklearn
import click
import pandas as pd
from pytest import param
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

#import sklearn.metrics as metrics

from sklearn.metrics import accuracy_score,v_measure_score,hamming_loss, f1_score, precision_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split    
from sklearn.model_selection import cross_val_score,cross_validate



@click.command()
@click.option(
    "-a",
    "--nested-cv",
    help="nested cross-validation",
    default=False,
    type=bool,
)
@click.option(
    "-d",
    "--decomposition",
    help="will be use TruncatedSVD",
    default=False,
    type=bool,
)
@click.option(
    "-r",
    "--n-components",
    default=3,
    help="n-components reduction",
    type=int,
    show_default=True,
)
@click.option(
    "-e",
    "--estimator",
    default="rf",
    type=click.Choice(["rf", "knn"]),
)
@click.option(
    "-p",
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
    help="n_estimators for RF / n_neighbors for KNN ",
    type=int,
    show_default=True,
)
@click.option(
    "-c",
    "--criterion",
    default="gini",
    type=click.Choice(["gini", "entropy"]),
    show_default=True,
)
@click.option(
    "-m",
    "--max-depth",
    default=None,
    type=int,
    show_default=True,
)
@click.option(
    "-cv",
    "--cv",
    default=5,
    type=int,
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
    default="uniform",
    type=click.Choice(["uniform", "distance"]),
    show_default=True,
)
@click.option(
    "-f",
    "--max-features",
    default="auto",
    type=click.Choice(["auto", "sqrt", "log2"], case_sensitive=False),
    show_default=True,
)
def train(
    estimator: str,
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    n_estimators: int,
    criterion: str,
    max_depth: int,
    cv: int,
    use_scaler: bool,
    weights:str,
    decomposition:bool,
    n_components: int,
    max_features: str,
    nested_cv: bool,
) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    # features_train, features_val, target_train, target_val = train_test_split(
    #   features, target, test_size=test_split_ratio)

    with mlflow.start_run():
        
        if nested_cv:
            cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)
            model = RandomForestClassifier(random_state=random_state)
            space = dict()
            space['n_estimators'] = [10, 50]
            space['max_features'] = ['auto', 'sqrt', 'log2', None] 
            space['criterion'] = ["gini", "entropy"]
            search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv_inner, refit=True)
            cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)
            scoring = {
                    "accuracy": make_scorer(accuracy_score),
                    "V_measure": make_scorer(v_measure_score),
                    "f1_score": make_scorer(f1_score, average='macro')
                    }
            scores = cross_validate(search, features, target, scoring=scoring, cv=cv_outer, n_jobs=-1, return_estimator = True)
            #click.echo(f"scores nested CV: {scores}.")
            scores_accuracy = scores['test_accuracy']
            scores_v_measure = scores['test_V_measure']
            scores_f1_macro = scores['test_f1_score']
            rf = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, refit=True)
            #click.echo(f"scores nested CV: {rf.get_params()}.")
            rf_param = rf.get_params()
            max_features = rf_param.get('max_features')
            click.echo(f"max_features : {max_features}.")
            rf_path = save_model_path
            rf_path = Path(rf_path.parent, f"{rf_path.stem}_rf_nested{rf_path.suffix}")
            dump(rf, rf_path)
        
        else:
            if estimator == "knn":
                if use_scaler:
                    knn_pipe = Pipeline([('sc', StandardScaler()),
                        ('knn', KNeighborsClassifier(n_neighbors=n_estimators, weights=weights))])
                    click.echo("used StandardScaler")
                    if decomposition:
                        knn_pipe = Pipeline([('sc', StandardScaler()),
                        ('svd', TruncatedSVD(n_components=n_components)),
                        ('knn', KNeighborsClassifier(n_neighbors=n_estimators, weights=weights))])
                        click.echo("used StandardScaler")
                        click.echo("used TruncatedSVD")
                elif decomposition:
                    knn_pipe = Pipeline([
                        ('svd', TruncatedSVD(n_components=n_components)),
                        ('knn', KNeighborsClassifier(n_neighbors=n_estimators, weights=weights))])
                    click.echo("used TruncatedSVD")
                else:
                    knn_pipe = Pipeline([
                        ('knn', KNeighborsClassifier(n_neighbors=n_estimators, weights=weights))])

                    # features_valid_scaled = scaler.transform(features_val)
                    click.echo("used TruncatedSVD")

                knn_pipe.fit(features, target)
                # y_predicted_knn = knn.predict(features_valid_scaled)
                scores_accuracy = cross_val_score(
                    knn_pipe,
                    features,
                    target,
                    scoring="accuracy",
                    cv=cv,
                    n_jobs=-1,
                )
                scores_v_measure = cross_val_score(
                    knn_pipe,
                    features,
                    target,
                    scoring="v_measure_score",
                    cv=cv,
                    n_jobs=-1,
                )
                scores_f1_macro = cross_val_score(
                    knn_pipe,
                    features,
                    target,
                    scoring="f1_macro",
                    cv=cv,
                    n_jobs=-1,
                )
                knn_path = save_model_path
                knn_path = Path(
                    knn_path.parent, f"{knn_path.stem}_knn{knn_path.suffix}"
                )
                print("save_model_path", save_model_path)
                print("knn_path", knn_path)
                dump(knn_pipe, knn_path)
                click.echo(f"Model is saved to {knn_path}.")
                
            elif estimator == "rf":

                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    criterion=criterion,
                    max_depth=max_depth,
                    max_features=max_features,
                    random_state=random_state,
                )
                if use_scaler:
                    rf_pipe = Pipeline([('rf', rf)])
                    click.echo("StandardScaler was ignored")
                    if decomposition:
                        rf_pipe = Pipeline([('svd', TruncatedSVD(n_components=n_components)),
                        ('rf', rf)])
                elif decomposition:
                    rf_pipe = Pipeline([
                        ('svd', TruncatedSVD(n_components=n_components)),
                        ('rf', rf)])
                else:
                    rf_pipe = Pipeline([
                        ('rf', rf)])

                scores_accuracy = cross_val_score(
                    rf_pipe, features, target, scoring="accuracy", cv=cv, n_jobs=-1
                )
                scores_v_measure = cross_val_score(
                    rf_pipe, features, target, scoring="v_measure_score", cv=cv, n_jobs=-1
                )
                scores_f1_macro = cross_val_score(
                    rf_pipe, features, target, scoring="f1_macro", cv=cv, n_jobs=-1
                )
                rf_path = save_model_path
                rf_path = Path(rf_path.parent, f"{rf_path.stem}_rf{rf_path.suffix}")
                dump(rf_pipe, rf_path)
                click.echo(f"Model is saved to {rf_path}.")
        
       # mlflow.sklearn.log_model(knn_path, "model", registered_model_name="KNN")

        mlflow.log_param("nested_cv", nested_cv)
        mlflow.log_param("_model", estimator)
        mlflow.log_param("n_ngb./n_est.", n_estimators)
        mlflow.log_param("criterion", criterion if estimator=="rf" else "-")
        mlflow.log_param("weights", weights if estimator=="knn" else "-")
        mlflow.log_param("use_scaler", use_scaler if (not nested_cv) and (estimator != "rf") else "not use")
        mlflow.log_param("max_depth", max_depth if estimator=="rf" else "-")
        mlflow.log_param("max_features", max_features if estimator=="rf" else "-")
        mlflow.log_param(
            "decomp.",
            "T.SVD (" + str(n_components) + ")" if decomposition and not nested_cv else "-",
        )
        mlflow.log_metric("accuracy", mean(scores_accuracy))
        mlflow.log_metric("v_measure", mean(scores_v_measure))
        mlflow.log_metric("f1_macro", mean(scores_f1_macro))


                

"""
@click.option(
    "-f",
    "--max-features",
    default="auto",
    type=int,
    show_default=True,
)
@click.option(
    "-f",
    "--max-features",
    default="auto",
    type=int,
    show_default=True,
)
"""
'''
cv_inner = KFold(n_splits=cv, shuffle=True, random_state=random_state)
rf = RandomForestClassifier(random_state=random_state)
param: dict[str, Any] = dict() 
param["n_estimators"] = [10, 50]  # [100, 300, 500]
param["max_features"] = ['auto', 'sqrt', 'log2'] 
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "V_measure": make_scorer(v_measure_score),
    "f1_score ": make_scorer(f1_score, average='macro')
    }

search = GridSearchCV(
    rf, param, scoring=scoring, refit="accuracy", n_jobs=1, cv=cv_inner,
)

cv_outer = KFold(n_splits=3, shuffle=True, random_state=1)
scores = cross_val_score(
    search, features, target, scoring="accuracy", cv=cv_outer, n_jobs=-1
)
result = search.fit(features, target)
#best_model = result.best_estimator_

# report performance

click.echo(f"Accuracy nested CV: {scores}.")
click.echo(f"Best param. nested CV: {result.best_params_}.")
'''