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
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
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
    mse_scores = cross_val_score(
        estimator,
        predictors,
        responses,
        cv=10,
        scoring='neg_mean_squared_error'
    )
    print 'Performance: %ss' % (datetime.datetime.utcnow() - start_time).total_seconds()
    print 'Mean MSE score and the 95% confidence interval:', mse_scores.mean(), '(+/-', mse_scores.std() * 2, ')'
    print 'Fitting'
    estimator.fit(predictors, responses)
    if hasattr(estimator, 'steps'):
        return estimator.steps[-1][-1].best_estimator_
    else:
        return estimator.best_estimator_

def get_model_by_name(model_name):
    params = {'C':[0.01, 0.1, 1, 10]}
    model = GridSearchCV(LinearSVR(epsilon=0), params, cv=10)
    return model

def test1(model_name, imputer_type):
    print 'Imputer (%s), No Scaling, No Feature Selection, %s' % (imputer_type, model_name)
    return evaluate_mse_score(get_model_by_name(model_name), imputer_type)

def test2(model_name, imputer_type):
    print 'Imputer (%s), No Scaling, Feature Selection F Regression 10%%, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('feature_selection', SelectPercentile(f_regression, percentile=10)),
        ('linear_model', get_model_by_name(model_name)),
    ])
    return evaluate_mse_score(estimator, imputer_type)

def test3(model_name, imputer_type):
    print 'Imputer (%s), No Scaling, Feature Selection F Regression 20%%, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('feature_selection', SelectPercentile(f_regression, percentile=20)),
        ('linear_model', get_model_by_name(model_name)),
    ])
    return evaluate_mse_score(estimator, imputer_type)

def test4(model_name, imputer_type):
    print 'Imputer (%s), Standardize, No Feature Selection, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('linear_model', get_model_by_name(model_name)),
    ])
    return evaluate_mse_score(estimator, imputer_type)

def test5(model_name, imputer_type):
    print 'Imputer (%s), Standardize, Feature Selection F Regression 10%%, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('feature_selection', SelectPercentile(f_regression, percentile=10)),
        ('linear_regression', get_model_by_name(model_name)),
    ])
    return evaluate_mse_score(estimator, imputer_type)

def test6(model_name, imputer_type):
    print 'Imputer (%s), Standardize, Feature Selection F Regression 20%%, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('feature_selection', SelectPercentile(f_regression, percentile=20)),
        ('linear_regression', get_model_by_name(model_name)),
    ])
    return evaluate_mse_score(estimator, imputer_type)

def test7(model_name, imputer_type):
    print 'Imputer (%s), No Scaling, Feature Selection Mutual Info Regression 20%%, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('feature_selection', SelectPercentile(mutual_info_regression, percentile=20)),
        ('linear_regression', get_model_by_name(model_name)),
    ])
    return evaluate_mse_score(estimator, imputer_type)

def test8(model_name, imputer_type):
    print 'Imputer (%s), Standardize, Feature Selection Mutual Info Regression 20%%, %s' % (imputer_type, model_name)
    estimator = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('feature_selection', SelectPercentile(mutual_info_regression, percentile=20)),
        ('linear_regression', get_model_by_name(model_name)),
    ])
    return evaluate_mse_score(estimator, imputer_type)


# 109. Linear SVR
for imputer_type in ('median', 'mode'):
    for func in (test1, test2, test3, test4, test5, test6, test7, test8):
        best_estimator = func('Linear SVR', imputer_type)
        print 'Parameters:'
        print 'C:', best_estimator.C

