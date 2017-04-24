"""Utility methods for making predictions.
"""

import numpy as np
import pandas as pd
from sklearn import metrics


def evaluate(targs, preds, name):

    print('=' * 80)
    print('EVALUATION (%s)' % name)
    print('-' * 80)

    is_continous = 'float' in preds.dtype.name
    if is_continous:
        loss = metrics.brier_score_loss(targs, preds)
        targs = targs.astype(np.float64)
    else:
        loss = metrics.mean_squared_error(targs, preds)

    print('loss: \t\t\t%s' % loss)
    print('area under curve:\t%s' % metrics.roc_auc_score(targs, preds))

    if not is_continous:
        print('accuracy:\t\t%s' % metrics.accuracy_score(targs, preds))
        print('precision:\t\t%s' % metrics.precision_score(targs, preds))
        print('recall:\t\t\t%s' % metrics.recall_score(targs, preds))
        print('f1:\t\t\t%s' % metrics.f1_score(targs, preds))

    print('=' * 80)
    print('')


def save_eviction_predictions(X, predictions, fname='data/prediction.csv'):
    """Write predictions to file.
    """
    challengeIDs = pd.DataFrame(data=range(1,4243), columns=['challengeID'])
    X = pd.concat([challengeIDs, X], axis=1)

    challengeIDs = X['challengeID'].values
    all_predictions = np.zeros((X.shape[0], 6))
    EVICTION_COL = 3
    all_predictions[:, EVICTION_COL] = predictions

    all_predictions = np.column_stack((challengeIDs, all_predictions))
    header = '"challengeID","gpa","grit","materialHardship","eviction",' \
             '"layoff","jobTraining"'

    # Sort by challengeID. This shouldn't be necessary, but just in case.
    all_predictions = all_predictions[all_predictions[:, 0].argsort()]

    # Why `comments` argument? See http://stackoverflow.com/a/17361181.
    BINARY = '%d'
    DECIMAL = '%10.10f'
    formats = [BINARY, DECIMAL, DECIMAL, DECIMAL, BINARY, DECIMAL, DECIMAL]
    np.savetxt(fname, all_predictions, header=header, delimiter=',',
               fmt=formats, comments='')


def gen_fname_from_model(Constructor, args, suffix=''):
    """Generate predictions filename from model_neg constructor and arguments.
    """
    fname = 'predict/predictions/prediction_%s' % Constructor.__name__
    if len(args) > 0:
        fname += '_'
        fname += '_'.join(['%s=%s' % (k, v) for k, v in args.items()])
    if suffix:
        fname += '_'
        fname += suffix
    fname += '.csv'
    return fname
