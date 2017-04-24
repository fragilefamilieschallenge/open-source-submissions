"""Predict eviction response variable.
"""

import data
from models import MODELS
from predict import test_runner


N_TESTS = 5
DIR = 'predict/results'
OVERSAMPLE = True


def full():
    dataset = data.load_eviction()
    fname = '%s/results_FULL_dataset_OVER.csv' % DIR
    test_runner.predict(MODELS, dataset, fname, N_TESTS, OVERSAMPLE)


def pca():
    dataset = data.load_eviction(dataset_type='pca')
    fname = '%s/results_PCA_dataset_OVER.csv' % DIR
    test_runner.predict(MODELS, dataset, fname, N_TESTS, OVERSAMPLE)


def rlogistic():
    dataset = data.load_eviction(dataset_type='rlogistic')
    fname = '%s/results_RLOGISTIC_dataset_OVER.csv' % DIR
    test_runner.predict(MODELS, dataset, fname, N_TESTS, OVERSAMPLE)


def handcrafted():
    dataset = data.load_eviction(dataset_type='handcrafted')
    fname = '%s/results_HANDCRAFTED_dataset_OVER.csv' % DIR
    test_runner.predict(MODELS, dataset, fname, N_TESTS, OVERSAMPLE)


if __name__ == '__main__':
    rlogistic()
    pca()
    handcrafted()
    full()
