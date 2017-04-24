"""Predict eviction response variable.
"""

import data
from models import MODELS
from predict import test_runner


N_TESTS = 5
DIR = 'predict/results'


def full():
    dataset = data.load_eviction()
    fname = '%s/results_FULL_dataset.csv' % DIR
    test_runner.predict(MODELS, dataset, fname, N_TESTS)


def pca():
    dataset = data.load_eviction(dataset_type='pca')
    fname = '%s/results_PCA_dataset.csv' % DIR
    test_runner.predict(MODELS, dataset, fname, N_TESTS)


def rlogistic():
    dataset = data.load_eviction(dataset_type='rlogistic')
    fname = '%s/results_RLOGISTIC_dataset.csv' % DIR
    test_runner.predict(MODELS, dataset, fname, N_TESTS)


def handcrafted():
    dataset = data.load_eviction(dataset_type='handcrafted')
    fname = '%s/results_HANDCRAFTED_dataset.csv' % DIR
    test_runner.predict(MODELS, dataset, fname, N_TESTS)


if __name__ == '__main__':
    rlogistic()
    pca()
    handcrafted()
    full()