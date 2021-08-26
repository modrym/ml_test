import os
import sys
import tarfile
import urllib

import numpy as np
import pandas as pd
from scipy.stats import expon, reciprocal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


DEBUG = print
# DEBUG = lambda *args: None

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join(os.getcwd(), "data", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)

    with tarfile.open(tgz_path) as fp:
        fp.extractall(path=housing_path)


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')

    if not os.path.isfile(csv_path):
        fetch_housing_data()

    return pd.read_csv(csv_path)


def shuffle_split(data):
    inc_cat = pd.cut(data['median_income'],
                     bins=[0, 1.5, 3.0, 4.5, 6., np.inf],
                     labels=list(range(1, 6)))

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_index, test_index = next(split.split(data, inc_cat))

    return data.loc[train_index].copy(), data.loc[test_index].copy()


def divide_set(data):
    return (data.drop('median_house_value', axis=1),
            data.median_house_value)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = (X[:, self.population_ix]
                                    / X[:, self.households_ix])

        if not self.add_bedrooms_per_room:
            return np.c_[X, rooms_per_household, population_per_household]

        bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]


def create_pipeline(labels):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])

    num_attribs = labels.copy()
    num_attribs.remove('ocean_proximity')
    cat_attribs = ['ocean_proximity']

    return ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs)
    ])


def main():
    housing = load_housing_data()
    train_set, test_set = shuffle_split(housing)
    X_train, y_train = divide_set(train_set)
    X_test, y_test = divide_set(test_set)

    full_pipeline = create_pipeline(list(X_train))

    X_train_p = full_pipeline.fit_transform(X_train)

    params = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

    estim = SVR()

    rand_search = RandomizedSearchCV(estim, params, cv=2, verbose=2,
                                     scoring='neg_mean_squared_error',
                                     return_train_score=True, n_iter=2)
    rand_search.fit(X_train_p, y_train)

    DEBUG(rand_search.best_params_)
    DEBUG(rand_search.best_score_)

    final_model = rand_search.best_estimator_

    X_test_p = full_pipeline.transform(X_test)

    DEBUG(final_model.score(X_test_p, y_test))

    return 0


if __name__ == '__main__':
    sys.exit(main())
