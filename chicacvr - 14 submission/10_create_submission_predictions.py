"""
Run predictions w/ the top 10 models against an unseen predictors set.
"""
import csv
import datetime
import pickle

import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.feature_selection import (f_regression, mutual_info_regression,
                                       SelectPercentile)
from sklearn.linear_model import (LinearRegression, Lars, Lasso, Ridge,
                                  ElasticNet, OrthogonalMatchingPursuitCV)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Load the train GPA map
reader = csv.reader(open('data/train.csv', 'r'))
next(reader)  # skip header row
train_gpa_map = {}
for row in reader:
    if row[1].lower() != 'na':
        train_gpa_map[int(row[0])] = row[1]

# Load the predictors and responses training data.
print 'Loading the training data from CSVs'
predictors_header = open
predictors_median = genfromtxt('data/predictors_median_training.csv', delimiter=',')
responses_median = genfromtxt('data/responses_median_training.csv', delimiter=',')
predictors_mode = genfromtxt('data/predictors_mode_training.csv', delimiter=',')
responses_mode = genfromtxt('data/responses_mode_training.csv', delimiter=',')

# Load the predictors and responses testing data.
print 'Loading the unseen testing data from CSVs'
unseen_predictors_median = genfromtxt('data/unseen_predictors_median.csv', delimiter=',')
unseen_predictors_mode = genfromtxt('data/unseen_predictors_mode.csv', delimiter=',')

# Load the training & testing headers and intersect the features they both have.
# Get a list of indexes of interesecting columns. We need this so that the shapes
# are matching for training and testing.
training_header = next(csv.reader(open('data/predictors_median.csv', 'r')))
testing_header = next(csv.reader(open('data/unseen_predictors_median.csv', 'r')))
training_set = set(training_header)
testing_set = set(testing_header)
intersection = training_set & testing_set
training_indexes = set(idx for idx, name in enumerate(training_header) if name in intersection)
testing_indexes = set(idx for idx, name in enumerate(testing_header) if name in intersection)

def _clean_data(matrix, indexes):
    new_rows = []
    for row in matrix:
        new_row = [cell for idx, cell in enumerate(row) if idx in indexes]
        new_rows.append(new_row)
    return new_rows


def make_prediction(estimator, imputer_type):
    if imputer_type not in ('median', 'mode'):
        raise ValueError('Wrong imputer type')

    # Training
    predictors = predictors_median if imputer_type == 'median' else predictors_mode
    responses = responses_median if imputer_type == 'median' else responses_mode

    start_time = datetime.datetime.utcnow()
    estimator.fit(_clean_data(predictors, training_indexes), responses)
    print 'Training: %ss' % (datetime.datetime.utcnow() - start_time).total_seconds()

    # Testing
    start_time = datetime.datetime.utcnow()
    unseen_predictors = unseen_predictors_median if imputer_type == 'median' else unseen_predictors_mode

    # Remove the header row and Challenge ID column for unseen predictors
    clean_unseen_predictors = _clean_data(unseen_predictors[1:], testing_indexes)

    # Predict the GPAs
    predicted = estimator.predict(clean_unseen_predictors)
    print 'Testing: %ss' % (datetime.datetime.utcnow() - start_time).total_seconds()

    # Create a map for Challenge ID to predicted GPA
    gpa_map = {}
    for idx, row in enumerate(unseen_predictors[1:]):
        gpa_map[int(row[-1])] = predicted[idx]

    return gpa_map


def fill_in_prediction_submission(gpa_map, out_filepath):
    # Load their prediction.csv, so that we can update it with our GPAs and
    # leave the rest as-is.
    reader = csv.reader(open('data/prediction.csv', 'r'))
    writer = csv.writer(open(out_filepath, 'w'))
    writer.writerow(next(reader))  # write header
    for row in reader:
        challenge_id = int(row[0])
        if challenge_id in train_gpa_map:
            row[1] = train_gpa_map[challenge_id]
        elif challenge_id in gpa_map:
            row[1] = gpa_map[challenge_id]
        writer.writerow(row)

    print 'Done!'


def test1(model, imputer_type, *args, **kwargs):
    print 'Imputer (%s), No Scaling, No Feature Selection, %s' % (imputer_type, model)
    return make_prediction(model, imputer_type, *args, **kwargs)

def test2(model, imputer_type, *args, **kwargs):
    print 'Imputer (%s), No Scaling, Feature Selection F Regression 10%%, %s' % (imputer_type, model)
    estimator = Pipeline([
        ('feature_selection', SelectPercentile(f_regression, percentile=10)),
        ('linear_model', model),
    ])
    return make_prediction(estimator, imputer_type, *args, **kwargs)

def test4(model, imputer_type, *args, **kwargs):
    print 'Imputer (%s), Standardize, No Feature Selection, %s' % (imputer_type, model)
    estimator = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('linear_model', model),
    ])
    return make_prediction(estimator, imputer_type, *args, **kwargs)

def test7(model, imputer_type, *args, **kwargs):
    print 'Imputer (%s), No Scaling, Feature Selection Mutual Info Regression 20%%, %s' % (imputer_type, model)
    estimator = Pipeline([
        ('feature_selection', SelectPercentile(mutual_info_regression, percentile=20)),
        ('linear_regression', model),
    ])
    return make_prediction(estimator, imputer_type, *args, **kwargs)

def test8(model, imputer_type, *args, **kwargs):
    print 'Imputer (%s), Standardize, Feature Selection Mutual Info Regression 20%%, %s' % (imputer_type, model)
    estimator = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('feature_selection', SelectPercentile(mutual_info_regression, percentile=20)),
        ('linear_regression', model),
    ])
    return make_prediction(estimator, imputer_type, *args, **kwargs)


# 1. Imputer (median), Standardize, Feature Selection Mutual Info Regression 20%, Elastic Net
gpa_map = test8(ElasticNet(alpha=0.0681579401299), 'median')
fill_in_prediction_submission(gpa_map, 'outputs/submission_predictions_model_1.csv')

# 2. Imputer (mode), Standardize, Feature Selection Mutual Info Regression 20%, Elastic Net
gpa_map = test8(ElasticNet(alpha=0.0901007856524), 'mode')
fill_in_prediction_submission(gpa_map, 'outputs/submission_predictions_model_2.csv')

# 3. Imputer (median), Standardize, No Feature Selection, Elastic Net
gpa_map = test4(ElasticNet(alpha=0.0966121191325), 'median')
fill_in_prediction_submission(gpa_map, 'outputs/submission_predictions_model_3.csv')

# 4. Imputer (mode), Standardize, No Feature Selection, Elastic Net
gpa_map = test4(ElasticNet(alpha=0.0966121191325), 'mode')
fill_in_prediction_submission(gpa_map, 'outputs/submission_predictions_model_4.csv')

# 5. Imputer (mode), Standardize, Feature Selection Mutual Info Regression 20%, Lasso
gpa_map = test8(Lasso(alpha=0.034078970065), 'mode')
fill_in_prediction_submission(gpa_map, 'outputs/submission_predictions_model_5.csv')

# 6. Imputer (median), Standardize, No Feature Selection, Lasso
gpa_map = test4(Lasso(alpha=0.0483060595662), 'median')
fill_in_prediction_submission(gpa_map, 'outputs/submission_predictions_model_6.csv')

# THESE RAN BEFORE
## 7. Imputer (median), Standardize, No Feature Selection, Orthogonal Matching Pursuit CV
#gpa_map = test4(OrthogonalMatchingPursuitCV(), 'median')
#fill_in_prediction_submission(gpa_map, 'outputs/submission_predictions_model_7.csv')
#
## 8. Imputer (median), No Scaling, No Feature Selection, Orthogonal Matching Pursuit CV
#gpa_map = test1(OrthogonalMatchingPursuitCV(), 'median')
#fill_in_prediction_submission(gpa_map, 'outputs/submission_predictions_model_8.csv')
#
## 9. Imputer (median), No Scaling, Feature Selection F Regression 10%, Decision Tree Regressor
#gpa_map = test2(DecisionTreeRegressor(max_depth=3), 'median')
#fill_in_prediction_submission(gpa_map, 'outputs/submission_predictions_model_9.csv')
#
## 10. Imputer (median), No Scaling, Feature Selection Mutual Info Regression 20%, Decision Tree Regressor
#gpa_map = test7(DecisionTreeRegressor(max_depth=3), 'median')
#fill_in_prediction_submission(gpa_map, 'outputs/submission_predictions_model_10.csv')















