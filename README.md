Домашнее задание №9 по курсу машинного обучения RS School.

В этом домашнем задании используется набор данных [Прогнозирование типа лесного покрова]
(https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset.

## Применение
Этот пакет позволяет обучать модель для прогнозирования типо лесного покрова(вида деревьев)вчетырех зонах дикой природы, расположенные в Национальном лесу Рузвельта на севере Колорадо. Признаками выступают данные предоставленые Геологической службой США и USFS (Лесная служба). 




 Clone this repository to your machine.
2. Download [Forest Cover Type Prediction] (https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset., save csv locally (default path is *data/*.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/): 
```
nox [-r]
```
Format your code with [black](https://github.com/psf/black) by using either nox or poetry:
```
nox -[r]s black
poetry run black src tests noxfile.py
```