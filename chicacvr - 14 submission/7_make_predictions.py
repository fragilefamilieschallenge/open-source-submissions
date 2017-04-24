"""
Evaluate top 10 models against a test set:
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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Load feature names.
with open('data/predictors_median.csv', 'r') as f:
    feature_names = next(csv.reader(f))

# Load codebook mappings
codebook_mappings = pickle.load(open('Codebook_Mappings.pck'))

# Load the predictors and responses training data.
print 'Loading the training data from CSVs'
predictors_header = open
predictors_median = genfromtxt('data/predictors_median_training.csv', delimiter=',')
responses_median = genfromtxt('data/responses_median_training.csv', delimiter=',')
predictors_mode = genfromtxt('data/predictors_mode_training.csv', delimiter=',')
responses_mode = genfromtxt('data/responses_mode_training.csv', delimiter=',')

# Load the predictors and responses testing data.
print 'Loading the testing data from CSVs'
predictors_median_test = genfromtxt('data/predictors_median_test.csv', delimiter=',')
responses_median_test = genfromtxt('data/responses_median_test.csv', delimiter=',')
predictors_mode_test = genfromtxt('data/predictors_mode_test.csv', delimiter=',')
responses_mode_test = genfromtxt('data/responses_mode_test.csv', delimiter=',')


def _filename_for_fig(title):
    return 'outputs/' + title.replace(' ', '_').replace('%', '').replace(',', '').replace('.', '_')

def make_residual_plot(predicted, responses, title):
    print 'Charting residual plot'
    plt.clf()
    plt.scatter(predicted, predicted - responses, c='b', s=40, alpha=0.5)
    plt.hlines(y=0, xmin=predicted.min(), xmax=predicted.max())
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.savefig(_filename_for_fig(title) + '_residual.png')

def make_prediction_error_plot(predicted, responses, title):
    print 'Charting prediction error plot'
    plt.clf()
    plt.scatter(predicted, responses, c='b', s=40, alpha=0.5)
    plt.plot([predicted.min(), predicted.max()], [responses.min(), responses.max()], 'k--', lw=3)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(_filename_for_fig(title) + '_prediction_error.png')

def make_prediction(estimator, imputer_type, title, use_coefs=True):
    if imputer_type not in ('median', 'mode'):
        raise ValueError('Wrong imputer type')

    # Training
    predictors = predictors_median if imputer_type == 'median' else predictors_mode
    responses = responses_median if imputer_type == 'median' else responses_mode

    start_time = datetime.datetime.utcnow()
    estimator.fit(predictors, responses)
    print 'Training: %ss' % (datetime.datetime.utcnow() - start_time).total_seconds()

    # Testing
    start_time = datetime.datetime.utcnow()
    predictors_test = predictors_median_test if imputer_type == 'median' else predictors_mode_test
    responses_test = responses_median_test if imputer_type == 'median' else responses_mode_test
    predicted = estimator.predict(predictors_test)
    print 'Testing: %ss' % (datetime.datetime.utcnow() - start_time).total_seconds()
    print 'R^2 score/coefficient of determination:', estimator.score(predictors_test, responses_test)
    print 'Mean squared error:', mean_squared_error(responses_test, predicted)

    # Make residual plot
    make_residual_plot(predicted, responses_test, title)

    # Make a prediction error plot
    make_prediction_error_plot(predicted, responses_test, title)

    # Get top 10 features, sorted by absolute coefficient value
    est = estimator.steps[-1][-1] if hasattr(estimator, 'steps') else estimator
    if use_coefs:
        top_features = sorted([
            (fname, f_coef)
            for fname, f_coef in zip(feature_names, est.coef_)
        ], key=lambda k: abs(k[1]), reverse=True)[:10]

        print '\nTop 10 features (based on absolute coefficient values):'
        for fname, coef in top_features:
            full_fname = codebook_mappings.get(fname)
            print full_fname or fname, coef
    else:
        top_features = sorted([
            (fname, f_imp)
            for fname, f_imp in zip(feature_names, est.feature_importances_)
        ], key=lambda k: k[1], reverse=True)[:10]

        print '\nTop 10 DT feature importances:'
        for fname, imp in top_features:
            full_fname = codebook_mappings.get(fname)
            print full_fname or fname, coef

    # Compute differences between actual and predicted values and sort them
    # in an ascending order (meaning the first values in the array are ones
    # where the actual value was high and the predicted value was low).
    diff = predicted - responses_test

    # Augment the diff with an ordinal index.
    diff = [(d, idx) for idx, d in enumerate(diff)]

    def process_diff(processed_diff):
        # Get predictors rows with the same indices and zip them with feature
        # names. This gives us the top 5 children who beat the odds.
        top_beating_the_odds = []
        for d, idx in processed_diff:
            top_beating_the_odds.append(
                zip(feature_names, predictors_test[idx])
            )

        # For each of these children, print their values of top 10 features (aka
        # questions).
        child_features = []
        for child_row in top_beating_the_odds:
            child_row = dict(child_row)  # transform into a dict for easier lookup
            child_features.append([
                (fname, child_row[fname]) for fname, coef in top_features
            ])

        # For each child row, print the short name of the feature, full name of
        # the feature, how they answered, predicted GPA and actual GPA
        for cnt, child_row in enumerate(child_features):
            print 'Child #%d' % (cnt+1)
            idx = processed_diff[cnt][1]
            print 'Predicted GPA:', predicted[idx], 'Actual GPA:', responses_test[idx]
            for fname, answer in child_row:
                full_fname = codebook_mappings.get(fname)
                print 'Short name:', fname, 'Full name:', full_fname, 'Answer:', answer
            print

    # Get the top 5 samples from the diff.
    # This is now a list of (difference, original ordinal index) tuples.
    print '\nTOP 5'
    top_diff = sorted(diff)[:5]
    process_diff(top_diff)

    # Get the worst 5 samples from the diff.
    # This is now a list of (difference, original ordinal index) tuples.
    print 'WORST 5'
    worst_diff =  sorted(diff, reverse=True)[:5]
    process_diff(worst_diff)

def test1(model, imputer_type, title):
    print title
    make_prediction(model, imputer_type, title)

def test2(model, imputer_type, title, use_coefs=True):
    print title
    estimator = Pipeline([
        ('feature_selection', SelectPercentile(f_regression, percentile=10)),
        ('linear_model', model),
    ])
    make_prediction(estimator, imputer_type, title, use_coefs=use_coefs)

def test4(model, imputer_type, title):
    print title
    estimator = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('linear_model', model),
    ])
    make_prediction(estimator, imputer_type, title)

def test7(model, imputer_type, title, use_coefs=True):
    print title
    estimator = Pipeline([
        ('feature_selection', SelectPercentile(mutual_info_regression, percentile=20)),
        ('linear_regression', model),
    ])
    make_prediction(estimator, imputer_type, title, use_coefs=use_coefs)

def test8(model, imputer_type, title):
    print title
    estimator = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('feature_selection', SelectPercentile(mutual_info_regression, percentile=20)),
        ('linear_regression', model),
    ])
    make_prediction(estimator, imputer_type, title)

# 1. Imputer (median), Standardize, Feature Selection Mutual Info Regression 20%, Elastic Net
test8(ElasticNet(alpha=0.0681579401299), 'median', title='1. Median, standardize, MI regression 20%, Elastic Net')

# 2. Imputer (mode), Standardize, Feature Selection Mutual Info Regression 20%, Elastic Net
test8(ElasticNet(alpha=0.0901007856524), 'mode', title='2. Mode, standardize, MI regression 20%, Elastic Net')

# 3. Imputer (median), Standardize, No Feature Selection, Elastic Net
test4(ElasticNet(alpha=0.0966121191325), 'median', title='3. Median, standardize, no feature selection, Elastic Net')

# 4. Imputer (mode), Standardize, No Feature Selection, Elastic Net
test4(ElasticNet(alpha=0.0966121191325), 'mode', title='4. Mode, standardize, no feature selection, Elastic Net')

# 5. Imputer (mode), Standardize, Feature Selection Mutual Info Regression 20%, Lasso
test8(Lasso(alpha=0.034078970065), 'mode', title='5. Mode, standardize, MI regression 20%, Lasso')

# 6. Imputer (median), Standardize, No Feature Selection, Lasso
test4(Lasso(alpha=0.0483060595662), 'median', title='6. Median, standardize, no feature selection, Lasso')

# THESE WERE RUN PREVIOUSLY
## 7. Imputer (median), Standardize, No Feature Selection, Orthogonal Matching Pursuit CV
#test4(OrthogonalMatchingPursuitCV(), 'median', title='7. Median, standardize, no feature selection, Orthogonal Matching Pursuit CV')
#
## 8. Imputer (median), No Scaling, No Feature Selection, Orthogonal Matching Pursuit CV
#test1(OrthogonalMatchingPursuitCV(), 'median', title='8. Median, no scaling, no feature selection, Orthogonal Matching Pursuit CV')
#
## 9. Imputer (median), No Scaling, Feature Selection F Regression 10%, Decision Tree Regressor
#test2(DecisionTreeRegressor(max_depth=3), 'median', title='9. Median, no scaling, f regression 10%, DTR', use_coefs=False)
#
## 10. Imputer (median), No Scaling, Feature Selection Mutual Info Regression 20%, Decision Tree Regressor
#test7(DecisionTreeRegressor(max_depth=3), 'median', title='10. Median, no scaling, MI regression 20%, DTR', use_coefs=False)
