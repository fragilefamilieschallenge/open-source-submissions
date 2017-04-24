"""Predict eviction response variable.
"""

import data
from predict import utils
from sklearn.svm import OneClassSVM


if __name__ == '__main__':

    dataset_pos = data.load_pos_eviction()
    dataset_neg = data.load_neg_eviction()
    dataset_all = data.load_eviction()

    # nu: The proportion of outliers we expect in our data.
    model_pos = OneClassSVM(kernel='linear', nu=0.9)
    model_pos.fit(dataset_pos.X_train)

    model_neg = OneClassSVM(kernel='linear', nu=0.1)
    model_neg.fit(dataset_neg.X_train)

    predictions_pos = model_pos.predict(dataset_all.X_train)
    predictions_neg = model_neg.predict(dataset_all.X_train)

    # +1 is inlier, -1 is outlier. We want those who are evicted, to be +1
    # and those who are not evicted to be 0.

    # Outliers, those evicted, to be 1.
    predictions_neg = (predictions_neg == -1).astype(int)

    # Inliers, those evicted, to be 1.
    predictions_pos = (predictions_pos == 1).astype(int)

    # Print results and mean squared error.
    utils.evaluate(dataset_all.y_train, predictions_pos, model_pos.__class__.__name__)
    utils.evaluate(dataset_all.y_train, predictions_neg, model_neg.__class__.__name__)
