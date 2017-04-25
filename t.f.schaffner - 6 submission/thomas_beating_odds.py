#!/usr/bin/python


import argparse
import numpy as np
import pandas as pd

from scipy.stats import fisher_exact


def fe(hh, hl, lh, ll):
    return fisher_exact([[hh, hl], [lh, ll]])[1]


def reconcile_tables(pred_fname, train_fname):
    pred = pd.read_csv(pred_fname)
    train = pd.read_csv(train_fname)

    train_cids = train.loc[:,'challengeID']

    pred = pred.loc[:,['challengeID', 'gpa', 'grit', 'materialHardship']]
    train = train.loc[:,['challengeID', 'gpa', 'grit', 'materialHardship']]

    pred = pred[pred.loc[:,'challengeID'].isin(train_cids)]
    train = train[train.loc[:,'challengeID'].isin(train_cids)]

    pred.index = pred.loc[:,'challengeID']
    train.index = train.loc[:,'challengeID']

    pred = pred.loc[:,['gpa', 'grit', 'materialHardship']]
    train = train.loc[:,['gpa', 'grit', 'materialHardship']]

    return pred, train


def calculate_deltas(pred, train):
    deltas = train - pred

    return deltas


def get_best_rank(deltas, num_ranked, category):
    selected = deltas.loc[:,category]
    nonnull = ~(selected.isnull())
    relevant = selected[nonnull]

    pairs = zip(relevant.index, relevant)

    sorted_pairs = sorted(pairs, key=lambda x: x[1])

    return [cid for cid, score in sorted_pairs[:num_ranked]]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--prediction", help="Path to predicted values", required=True)
    parser.add_argument("-t", "--train", help="Path to training values", required=True)
    parser.add_argument("-n", "--num_ranked", help="Number of samples to collect for ranked lists", required=True, type=int)

    options = parser.parse_args()

    run(options)


def run(options):
    prediction, training = reconcile_tables(options.prediction, options.train)

    deltas = calculate_deltas(prediction, training)
    best_gpa = get_best_rank(deltas, options.num_ranked, 'gpa')
    best_grit = get_best_rank(deltas, options.num_ranked, 'grit')
    best_hard = get_best_rank(deltas, options.num_ranked, 'materialHardship')

    print best_gpa
    print best_grit
    print best_hard

    print set(best_gpa).intersection(set(best_grit)).intersection(set(best_hard))


if __name__ == "__main__":
    main()