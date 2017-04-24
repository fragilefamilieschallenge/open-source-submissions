"""General preprocessing script for all methods.
"""

import numpy as np
import pandas as pd
from preprocess.impute import DataFrameImputer
from preprocess.factorize import factorize


def clean():
    """Convert raw data by:

    1. Imputing missing values.
    2. Dropping columns with all NaN values.
    3. Encoding categorical values with numeric labels.
    """
    X = pd.read_pickle('data/private/X_raw.p')
    y_train = pd.DataFrame.from_csv('data/private/y_train_raw.csv')
    print('Initial:\t\t\t\t(%s, %s)' % X.shape)

    # Impute data. Numeric values are filled with mean; categorical values are
    # filled with mode.
    X = DataFrameImputer().fit_transform(X)

    # Drop columns which are entirely NaN.
    if X.isnull().values.any():
        X = X.dropna(how='all', axis=1)
    print('After dropping NaN columns:\t\t(%s, %s)' % X.shape)

    # Encode categorical variables with numeric labels.
    X = factorize(X)

    # Sort predictions by challengeID and then drop that column. We do not
    # want to make predictions based on it.
    X = X.sort_values(by='challengeID')
    X = X.drop('challengeID', 1)
    # The labels (y_train) is already properly sorted.
    print('After dropping challenge ID column:\t(%s, %s)' % X.shape)

    print('Final:\t\t\t\t\t(%s, %s)' % X.shape)
    X.to_pickle('data/private/X_cleaned.p')
    y_train.to_pickle('data/private/y_train.p')


if __name__ == '__main__':
    clean()
