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
[../data/], {Path}. Путь к database,  и место сохранения модели.

```sh
-s, --save-model-path  
```
[data/model.joblib], {FILE}. В зависимости от выбранного алгоритма к имени файла добавиться суффикс (*knn, *rf, *rf_nested)
```

```sh
-report, --prfl-report  
```
[False], {BOOLEAN}. Создаст в корне директория файл EDAreport.html - стандартный отчет pandas_profiling

```sh
--random-state  
```
[42], {INTEGER}. Параметр генератора случайных значений

```sh
--test-split-ratio  
```
[0.2], {FLOAT}. Коэффициент разбиения dataset (не используется в данной реализации)

```sh
  -fe, --fe
```  
[False], {BOOLEAN}. Применить feature engineering. Будут удалены 2 приназа с дисперсией=0, и из двух признаков "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology" составлен один "Eucl_dist_Hydr" как евклидово расстояние.


```sh
  -d, --decomposition
```  
[False], {BOOLEAN}. Использовать уменьшение размерности TruncatedSVD

```sh
  -r, --n-components 
```
[3], {INTEGER} количество компонент после уменьшения размерности TruncatedSVD. оганичение [1, database.shape[1]]

```sh
  -sc, --use-scaler 
```
[True], {BOOLEAN} Флаг применения StandardScaler для алгоритма KNeighborsClassifier.

```sh
-e, --estimator
```
[rf], {rf,knn}. Выбор алгоритма классификации "rf" - RandomForestClassifier, "knn" - KNeighborsClassifier. 
Данный ключ является корневым, в зависимости от его выбора будут учитываться или не учитываться остальные ключи.

```sh
-n, --n-estimators
```
[100], {INTEGER}. Значение n-estimators при выборе алгоритма RandomForestClassifier или n_neighbors при KNeighborsClassifier 
Хотя при выборе KNN, n_neighbors: [1, database.shape[0]], к выбору этого параметра необходимо подходить взвешано.

```sh
-w, --weights
```
[uniform], {uniform|distance}. Выбор весовая функции для алгоритма KNeighborsClassifier.

```sh
-с, --criterion
```
[gini], {gini|entropy}. При выборе алгоритма RandomForestClassifier возможен выбор алгоритма рачщепления

```sh
-f, --max-features
```
[auto], {auto|sqrt|log2}. Количесвто функций при разделении -параметр выбора лучшего сплита для алгоритма RandomForestClassifier

```sh
-m, --max-depth
```
[None], {INTEGER}. Максимальная глубина дерева для алгоритма RandomForestClassifier. Если None, то узлы расширяются до тех пор, пока все листья не станут чистыми или пока все листья не будут содержать выборок меньше, чем min_samples_split.

```sh
-cv, --cv
```
[5], {INTEGER}. Количество сгибов при выполнении cross_val_score. Используется при подсчете метрик при RandomForestClassifier или KNeighborsClassifier 

```sh
-a, --nested-cv
```
[False], {BOOL}. Запустить алгоритм подбора гиперпараметров и расчета метрик NestedCV. Параметры алгоритма, пока задаются в коде (cv_inner = 3, n_estimators = [50, 100, 250,500], max_features = ['auto', 'sqrt', 'log2'], cv_outer = 5 ). Применяемый алгоритм классификации - RandomForestClassifier. 

6. Запустите MLflow UI, чтобы увидеть результаты,выполненных вами эксперементов:
```sh
poetry run mlflow ui
```
После запуска, результат можно увидеть в браузере по адресу: http://127.0.0.1:5000/


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
Результат работы модуля black
![black_result](https://github.com/wert1999/9-Task-ml-intro/black.png)

Результат работы модуля mypy
![mypy_result](https://github.com/wert1999/9-Task-ml-intro/mypy.png)
