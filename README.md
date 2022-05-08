Домашнее задание №9 по курсу машинного обучения RS School.

В этом домашнем задании используется набор данных [Прогнозирование типа лесного покрова]
(https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset.

## Применение
Этот пакет позволяет обучать модель для прогнозирования типо лесного покрова(вида деревьев)вчетырех зонах дикой природы, расположенные в Национальном лесу Рузвельта на севере Колорадо. Признаками выступают данные предоставленые Геологической службой США и USFS (Лесная служба). 


1. Клонируйте репозиторий на свой компьютер
2. Загрузите [Forest Cover Type Prediction] (https://www.kaggle.com/competitions/forest-cover-type-prediction/data). Сохраните csv локально  (путь по умолчанию *data/*.csv* в корне репозитория).
3. Убедитесь в наличие Python 3.9 и [Poetry](https://python-poetry.org/docs/) Poetry 1.1.13.
4. Установите зависимости проекта (*запустите эту и следущие команды из корневой папки клонированного репозитория*):
```sh
poetry install --no-dev
```
5. Запустите train с ключом --help, чтобы посмотреть параметры коммандной строки:
```sh
poetry run train --help
```
Подробнее о ключах запуска (в [] скобках указан ключ по умолчанию, {} скобках тип ключа).

```sh
-p, --dataset-path  
```
[../data/], {Path}. Пусть к database,  и место сохранения модели.
```sh
-report  
```
[False], {BOOLEAN}. Создаст в корне директория файл EDAreport.html - стандартный отчет pandas_profiling
```sh
-e, --estimator
```
[rf], {rf,knn}. Выбор алгоритма классификации "rf" - RandomForestClassifier, "knn" - KNeighborsClassifier
```sh
  -d, --decomposition
```  
[False], {BOOLEAN}. использовать уменьшение размерности TruncatedSVD
```sh
  -r, --n-components INTEGER      n-components reduction  [default: 3]
```
[3], {INTEGER} количество компонент после уменьшения размерности TruncatedSVD. оганичение [1, database.shape[1]]


You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
```
6. Запустите MLflow UI, чтобы увидеть результаты,выполненных вами эксперементов:
```sh
poetry run mlflow ui
```

## Разработчикам

Код этого репозитория может быть оттестирован, отформатирован, используюя "black", и проверен на соответствие типов с помощью "mypy",перед тем, как он будет помещен в репозиторий.

Установите все требования (включая требования разработчика) к среде poetry:
```
poetry install
```
Теперь вы можете использовать инструменты разработчика, например pytest:
```
poetry run pytest
```
Удобнее, чтобы запускать все сеансы тестирования и форматирования одной командой, установите и используйте [nox](https://nox.thea.codes/ru/stable/):
```
nox [-r]
```
Отформатируйте свой код с помощью [black](https://github.com/psf/black) используя [nox] или [poetry]:
```
nox -[r]s black
poetry run black src tests noxfile.py
```