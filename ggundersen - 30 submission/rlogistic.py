"""Preprocessing script to find most stable features, i.e. those that are
selective across many randomized trials.
"""

import pandas as pd
from sklearn.linear_model import RandomizedLogisticRegression
import data


if __name__ == '__main__':
    dataset = data.load_eviction()
    descriptions = pd.read_pickle('data/private/feature_codes_to_names.pck')

    print('Data loaded.')

    rlogistic = RandomizedLogisticRegression(normalize=True)
    rlogistic.fit(dataset.X_train, dataset.y_train)

    print('Model fitted.')

    features = sorted(zip(map(lambda x: round(x, 4), rlogistic.scores_),
                          dataset.X_train.columns))
    print('Number of features:\t\t%s' % len(features))

    nonzero_features = [(score, code) for score, code in features if score > 0]
    print('Number of nonzero features:\t%s' % len(nonzero_features))

    columns = []
    for score, code in nonzero_features:
        columns.append(code)
        print('-' * 80)
        print(score)
        if code in descriptions:
            print('%s: %s' % (code, descriptions[code]))
        else:
            print(code)

    dataset.X[columns].to_pickle('data/private/X_rlogistic.p')
