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
    return estimator

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


# 2. Imputer (mode), Standardize, Feature Selection Mutual Info Regression 20%, Elastic Net CV
estimator = test8('Elastic Net CV', 'mode')
print 'Parameters:'
print 'Alpha:', estimator.steps[-1][-1].alpha_
print 'L1 ratio:', estimator.steps[-1][-1].l1_ratio_

# 1. Imputer (median), Standardize, Feature Selection Mutual Info Regression 20%, Elastic Net CV
estimator = test8('Elastic Net CV', 'median')
print 'Parameters:'
print 'Alpha:', estimator.steps[-1][-1].alpha_
print 'L1 ratio:', estimator.steps[-1][-1].l1_ratio_

# 3. Imputer (median), Standardize, No Feature Selection, Elastic Net CV
estimator = test4('Elastic Net CV', 'median')
print 'Parameters:'
print 'Alpha:', estimator.steps[-1][-1].alpha_
print 'L1 ratio:', estimator.steps[-1][-1].l1_ratio_

# 4. Imputer (mode), Standardize, No Feature Selection, Elastic Net CV
estimator = test4('Elastic Net CV', 'mode')
print 'Parameters:'
print 'Alpha:', estimator.steps[-1][-1].alpha_
print 'L1 ratio:', estimator.steps[-1][-1].l1_ratio_

# 5. Imputer (mode), Standardize, Feature Selection Mutual Info Regression 20%, Lasso CV
estimator = test8('Lasso CV', 'mode')
print 'Parameters:'
print 'Alpha:', estimator.steps[-1][-1].alpha_

# 6. Imputer (median), Standardize, No Feature Selection, Lasso CV
estimator = test4('Lasso CV', 'median')
print 'Parameters:'
print 'Alpha:', estimator.steps[-1][-1].alpha_

# Worst ones

# Median, No Scaling, F Reg 10%, Linear Regression
estimator = test2('Linear Regression', 'median')

# Mode, No Scaling, F Reg 10%, Linear Regression
estimator = test2('Linear Regression', 'mode')

# Mode, Standardize, MI 20%, Linear Regression
estimator = test8('Linear Regression', 'mode')

# Median, Standardize, No Feature Selection, Linear Regression
estimator = test4('Linear Regression', 'median')

# Mode, Standardize, No Feature Selection, Linear Regression
estimator = test4('Linear Regression', 'mode')
predictors = predictors_mode
responses = responses_mode
predicted = cross_val_predict(
    estimator,
    predictors.copy(),
    responses.copy(),
    cv=10,
)
print 'Charting residual plot'
plt.clf()
plt.scatter(predicted, predicted - responses, c='b', s=40, alpha=0.5)
plt.hlines(y=0, xmin=predicted.min(), xmax=predicted.max())
plt.title('Mode, Standardize, No Feature Selection, Linear Regression')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.savefig('outputs/worst_lin_reg_residual.png')


################################################################################

# No need to run the ones below, because they don't have any hyperparameters

# 7. Imputer (median), Standardize, No Feature Selection, Orthogonal Matching Pursuit CV
#estimator = test4('Orthogonal Matching Pursuit CV', 'median')

# 8. Imputer (median), No Scaling, No Feature Selection, Orthogonal Matching Pursuit CV
#estimator = test1('Orthogonal Matching Pursuit CV', 'median')

# 9. Imputer (median), No Scaling, Feature Selection F Regression 10%, Decision Tree Regressor
# estimator = test2('Decision Tree Regressor', 'median')

# 10. Imputer (median), No Scaling, Feature Selection Mutual Info Regression 20%, Decision Tree Regressor
# estimator = test7('Decision Tree Regressor', 'median')


