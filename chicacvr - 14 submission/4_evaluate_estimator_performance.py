"""
Combination of an Imputer (median/mode), Scaling (no scaling/standardize),
Feature Selection (no selection, f regression 10%, f regression 20%, rfecv),
and Linear Model (linear regression, LARS, OMP, Lasso, Ridge, Elastic Net).
"""
import datetime
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.feature_selection import (f_regression, mutual_info_regression,
                                       SelectPercentile)
from sklearn.linear_model import (LinearRegression, LarsCV, LassoCV, RidgeCV,
                                  ElasticNetCV, OrthogonalMatchingPursuitCV)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Load the predictors and responses training data.
print 'Loading the data from CSVs'
predictors_median = genfromtxt('data/predictors_median_training.csv', delimiter=',')
responses_median = genfromtxt('data/responses_median_training.csv', delimiter=',')
predictors_mode = genfromtxt('data/predictors_mode_training.csv', delimiter=',')
responses_mode = genfromtxt('data/responses_mode_training.csv', delimiter=',')

def evaluate_mse_score(estimator, imputer_type):
    """Evaluate a given estimator and print information about its
    accuracy and performance.
    """
    if imputer_type not in ('median', 'mode'):
        raise ValueError('Wrong imputer type')

    predictors = predictors_median if imputer_type == 'median' else predictors_mode
    responses = responses_median if imputer_type == 'median' else responses_mode
    start_time = datetime.datetime.utcnow()
    mse_score = cross_val_score(
        estimator,
        predictors.copy(),
        responses.copy(),
        cv=10,
        scoring='neg_mean_squared_error'
    )
    print 'Mean MSE score:', mse_score.mean()
    print 'Performance: %ss' % (datetime.datetime.utcnow() - start_time).total_seconds()

def get_model_by_name(model_name):
    return {
        'Linear Regression': LinearRegression(),
        'Lars CV': LarsCV(cv=10),
        'Lasso CV': LassoCV(cv=10),
        'Ridge CV': RidgeCV(cv=10),
        'Elastic Net CV': ElasticNetCV(cv=10),
        'Orthogonal Matching Pursuit CV': OrthogonalMatchingPursuitCV(cv=10),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=3),
    }[model_name]

def test1(model_name, imputer_type):
    print 'Imputer (%s), No Scaling, No Feature Selection, %s' % (imputer_type, model_name)
    evaluate_mse_score(get_model_by_name(model_name), imputer_type)

def test2(model_name, imputer_type):
    print 'Imputer (%s), No Scaling, Feature Selection F Regression 10%%, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('feature_selection', SelectPercentile(f_regression, percentile=10)),
        ('linear_model', get_model_by_name(model_name)),
    ])
    evaluate_mse_score(estimator, imputer_type)

def test3(model_name, imputer_type):
    print 'Imputer (%s), No Scaling, Feature Selection F Regression 20%%, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('feature_selection', SelectPercentile(f_regression, percentile=20)),
        ('linear_model', get_model_by_name(model_name)),
    ])
    evaluate_mse_score(estimator, imputer_type)

def test4(model_name, imputer_type):
    print 'Imputer (%s), Standardize, No Feature Selection, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('linear_model', get_model_by_name(model_name)),
    ])
    evaluate_mse_score(estimator, imputer_type)

def test5(model_name, imputer_type):
    print 'Imputer (%s), Standardize, Feature Selection F Regression 10%%, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('feature_selection', SelectPercentile(f_regression, percentile=10)),
        ('linear_regression', get_model_by_name(model_name)),
    ])
    evaluate_mse_score(estimator, imputer_type)

def test6(model_name, imputer_type):
    print 'Imputer (%s), Standardize, Feature Selection F Regression 20%%, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('feature_selection', SelectPercentile(f_regression, percentile=20)),
        ('linear_regression', get_model_by_name(model_name)),
    ])
    evaluate_mse_score(estimator, imputer_type)

def test7(model_name, imputer_type):
    print 'Imputer (%s), No Scaling, Feature Selection Mutual Info Regression 20%%, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('feature_selection', SelectPercentile(mutual_info_regression, percentile=20)),
        ('linear_regression', get_model_by_name(model_name)),
    ])
    evaluate_mse_score(estimator, imputer_type)

def test8(model_name, imputer_type):
    print 'Imputer (%s), Standardize, Feature Selection Mutual Info Regression 20%%, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('feature_selection', SelectPercentile(mutual_info_regression, percentile=20)),
        ('linear_regression', get_model_by_name(model_name)),
    ])
    evaluate_mse_score(estimator, imputer_type)

# Run faster tests first
for model_name in (
    'Linear Regression',
    'Lars CV',
    'Ridge CV',
    'Elastic Net CV',
    'Orthogonal Matching Pursuit CV',
    'Lasso CV',
):
    for imp_type in ('median', 'mode'):
        test1(model_name, imp_type)
        test2(model_name, imp_type)
        test3(model_name, imp_type)
        test5(model_name, imp_type)
        test6(model_name, imp_type)

# Run a subset of tests for the Decision Tree Regressor
model_name = 'Decision Tree Regressor'
for imp_type in ('median', 'mode'):
    test2(model_name, imp_type)
    test3(model_name, imp_type)
    test5(model_name, imp_type)
    test6(model_name, imp_type)
    test7(model_name, imp_type)
    test8(model_name, imp_type)

# Run slower tests last
for model_name in (
    'Linear Regression',
    'Lars CV',
    'Ridge CV',
    'Elastic Net CV',
    'Orthogonal Matching Pursuit CV',
    'Lasso CV',
):
    for imp_type in ('median', 'mode'):
        test4(model_name, imp_type)
        test7(model_name, imp_type)
        test8(model_name, imp_type)

