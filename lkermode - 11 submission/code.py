import csv
import sys
from MissingDataScript import fillMissing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
from sklearn.externals import joblib
import time

csv.field_size_limit(sys.maxsize)

class Dataset:
  def __init__(self):
    self.training_results = None
    self.training_set = None
    self.training_set_annotated = None
    self.full_set = None
    self.full_set_annotated = None
    # meta
    self.predictions_folder = 'predictions'
    self._setup()
    if ((self.training_set is None) or (self.training_set_annotated is None) or (self.full_set is None) or (self.full_set_annotated is None) or (self.training_results is None)):
      raise NotImplementedError('users must set variables: training_results, training_set, training_set_annotated, full_set, full_set_annotated')
    print('Setup complete.')

  def _setup(self):
    raise NotImplementedError('users must define "_setup" to use this class')


# Uncomment this line to run the missing data script on the raw BACKGROUND_FILE and TRAINING_FILE
# --------------------------------------------
# fillMissing(BACKGROUND_FILE, BACKGROUND_FILLED) 
# fillMissing(TRAINING_FILE, TRAINING_FILLED)
# --------------------------------------------

class FragileFams(Dataset):
  def _setup(self):
    PATH = 'fragilefamilieschallenge/'
    BACKGROUND_FILE = PATH + 'background.csv'
    PREDICTION_FILE = PATH + 'prediction.csv'
    TRAINING_FILE = PATH +'train.csv'
    BACKGROUND_FILLED = PATH+'background_filled.csv'
    TRAIN_FILLED = PATH+'train_filled.csv'

    if not (os.path.isfile(TRAIN_FILLED) and os.path.isfile(BACKGROUND_FILLED)):
      raise ValueError('train_filled and background_filled do not exist.')

    background_raw = pd.read_csv(BACKGROUND_FILLED, low_memory=False)
    training_raw = pd.read_csv(TRAIN_FILLED, low_memory=False)
    train_results = pd.read_csv(TRAINING_FILE)
    print('Loaded files.')

    def in_training(x):
      cid = background_raw.iloc[x]['challengeID']
      return cid in training_raw['challengeID'].as_matrix()

    # get background values where challengeID is in training
    background_in_training = background_raw.select(in_training)

    training_set_annotated = background_in_training.select_dtypes(exclude=['object']) # drop inconsistent values
    self.training_set_annotated = training_set_annotated.sort_values('challengeID')
    self.training_set = training_set_annotated.drop(['challengeID', 'idnum'], axis=1)

    full_set_annotated = background_raw.select_dtypes(exclude=['object'])
    self.full_set_annotated = full_set_annotated.sort_values('challengeID')
    self.full_set = full_set_annotated.drop(['challengeID', 'idnum'], axis=1)

    # training values drop challengeID before going into the classifier, as they now line up with bit.
    self.training_results = training_raw.drop('challengeID', axis=1)
    self.training_results_annotated = training_raw 


    ev_fs = eviction
    lay_fs = layoff
    job_fs = jobTraining
    
    clf_eviction = clf_dfs[3]['eviction'][ev_fs]
    clf_layoff = clf_dfs[3]['layoff'][lay_fs]
    clf_job = clf_dfs[3]['jobTraining'][job_fs]
    _start = time.process_time()
    eviction_preds = clf_eviction.predict_proba(data.full_set.filter(fs_df['eviction'][ev_fs]).values)[:,1]
    layoff_preds = clf_layoff.predict_proba(data.full_set.filter(fs_df['layoff'][lay_fs]).values)[:,1]
    job_preds = clf_job.predict_proba(data.full_set.filter(fs_df['jobTraining'][job_fs]).values)[:,1]
    _end = time.process_time()
    
    results = pd.DataFrame()
    results['eviction'] = eviction_preds
    results['layoff'] = layoff_preds
    results['jobTraining'] = job_preds
    print('Predictions complete in', _end - _start, 'seconds.')
    return results


""" Map on a dataframe of binary_cols with fn(item, row, col), create new dataframe """
def df_map(fn, df):
    binary_cols = ['eviction', 'layoff', 'jobTraining']
    items = pd.DataFrame(columns=binary_cols)
    for index in df.index:
        row = []
        for col in df.columns:
            item = df.loc[index,col]
            new_item = fn(item, index, col)
            row.append(new_item)
        items.loc[index] = row
    return items

""" Get coefficients for a dataframe of classifiers """
def get_coefs(df):
  return df_map(lambda clf,x,y:clf.coef_)

""" Get intercepts for a dataframe of classifiers """
def get_intercepts(df):
  return df_map(lambda clf,x,y:clf.coef_)


## ------------------------------------------------------------------------------------------------------ ##
##   GENERATION
## ------------------------------------------------------------------------------------------------------ ##
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, RFE

data = FragileFams()
binary_cols = ['eviction', 'layoff', 'jobTraining']
feature_selections = [
    ('k200', SelectKBest(chi2, 200)),
    ('k2000', SelectKBest(chi2, 2000)),
    ('k5000', SelectKBest(chi2, 5000)),
    ('variance_threshold', VarianceThreshold(threshold=(.8 * (1 - .8))))
#     ('f_selection', RFE( SVR(kernel='linear'), 5, step=1 ))
]

# save predictions on the full_set to a .txt file in predictions.
def selected_features_df():
    df = pd.DataFrame(columns=binary_cols)
    for name, _selection in feature_selections:
        row = []
        for column in binary_cols:
            k_values = _selection.fit(data.training_set_annotated.values, data.training_results[column].values)
            row.append(data.training_set_annotated.columns[k_values.get_support()])
        df.loc[name] = row
    return df
        
fs_df = selected_features_df()
fs_df.loc['all'] = [data.training_set_annotated.columns for _ in  range(0,3)]
fs_df.to_pickle('dataframes/feature_selections.pkl')
print('Features selections dataframe generated.')

def df_map(fn, df):
    items = pd.DataFrame(columns=binary_cols)
    for index in df.index:
        row = []
        for col in df.columns:
            item = df.loc[index,col]
            new_item = fn(item, index, col)
            row.append(new_item)
        items.loc[index] = row
    return items

""" Fit a classifier on all of the values in a fs_df, using different features from their selections. """
def fit_clfs(_clf, debug=False, features_df=fs_df):
    def fit_training(features, row, col):
        if (debug):
            _begin = time.process_time()
            print('Training classifier for {0} with {1} features selected...'.format(col, row))
        clf = _clf().fit(data.training_set.filter(items=features), data.training_results[col].values)
        if (debug):
            _end = time.process_time()
            _total_time = _end - _begin
            print('-- time to fit: {0}'.format(_total_time))
        return clf

    if (debug):
        begin = time.process_time()
    new_df = df_map(fit_training, features_df)
    if (debug):
        end = time.process_time()
        total_time = end - begin
        print('Training complete in {0} 10th of seconds'.format(total_time))
    return new_df

def get_intercepts(item, row, col):
    return item.intercept_

def get_coefs(item, row, col):
    return item.coef_

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier


## Fast Classifiers
# save classifier matrices. NB: generating a new one each time as I don't know how the internals of fitting works.
def rand_forest():
    return RandomForestClassifier()
def k_nearest_neighbors():
    return KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')
def gnb():
    return GaussianNB()
def mnb():
    return MultinomialNB()
def bnb():
    return BernoulliNB()
def dt():
    return DecisionTreeClassifier()

rf_clfs = fit_clfs(rand_forest, debug=True)
# knn_clfs = fit_clfs(k_nearest_neighbors, debug=True)
gnb_clfs = fit_clfs(gnb, debug=True)
mnb_clfs = fit_clfs(mnb, debug=True)
bnb_clfs = fit_clfs(bnb, debug=True)
dt_clfs = fit_clfs(dt, debug=True)

rf_clfs.to_pickle('dataframes/rf_classifiers.pkl')
# knn_clfs.to_pickle('dataframes/knn_classifiers.pkl')
gnb_clfs.to_pickle('dataframes/gnb_classifiers.pkl')
mnb_clfs.to_pickle('dataframes/mnb_classifiers.pkl')
bnb_clfs.to_pickle('dataframes/bnb_classifiers.pkl')
dt_clfs.to_pickle('dataframes/dt_classifiers.pkl')

print('Classifier matrices generated.')

onlyk200 = fs_df.loc[['k200']]
all_fs = fs_df.drop('all')

## Logistic Regression
def lr():
    return LogisticRegression()

log_reg_clfs = fit_clfs(lr, debug=True, features_df=all_fs)
# lr_clfs.to_pickle('dataframes/logistic_reg_classifiers(wo-all).pkl')
print('Logistic Regression: training complete.')

from sklearn.linear_model import LinearRegression

def linear_reg():
    return LinearRegression()

linear_reg_clfs = fit_clfs(linear_reg, debug=True)
# linear_reg_clfs.to_pickle('dataframes/linear_reg_classifiers.pkl')
print('Linear Regression: training complete.')

def log_reg_l1():
    return LogisticRegression(penalty='l1')

log_reg_l1_clfs = fit_clfs(log_reg_l1, debug=True, features_df=all_fs)
# log_reg_l1_clfs.to_pickle('dataframes/log_reg_l1_classifiers.pkl')
print('Logistic Regression L1: training complete.')

from sklearn.linear_model import Ridge
def ridge_reg():
    return Ridge(alpha=0.5)
ridge_reg_clfs = fit_clfs(ridge_reg, debug=True)
# ridge_reg_clfs.to_pickle('dataframes/ridge_reg_classifiers.pkl')
print('Ridge Regression: training complete.')

from sklearn.linear_model import Lasso
def lasso_reg():
    return Lasso(alpha=0.1)
lasso_reg_clfs = fit_clfs(lasso_reg, debug=True)
# lasso_reg_clfs.to_pickle('dataframes/lasso_reg_classifiers.pkl')
print('Lasso Regression: training complete.')

from sklearn.linear_model import LassoLars
def larslasso():
    return LassoLars(alpha=.2)
larslasso_clfs = fit_clfs(larslasso, debug=True)
# larslasso_clfs.to_pickle('dataframes/larslasso_classifiers.pkl')
print('LassoLars: training complete.')

## ------------------------------------------------------------------------------------------------------ ##
##   PREDICTION
## ------------------------------------------------------------------------------------------------------ ##

# df_folder = 'dataframes'
# fs_file = 'dataframes/feature_selections.pkl'
# files_to_load = [
#   'rf_classifiers.pkl',
#   'gnb_classifiers.pkl',
#   'bnb_classifiers.pkl',
#   'mnb_classifiers.pkl',
#   'dt_classifiers.pkl',
#   'linear_reg_classifiers.pkl',
#   'logistic_reg_classifiers(wo-all).pkl',
#   'log_reg_l1_classifiers.pkl',
#   'ridge_reg_classifiers.pkl',
#   'lasso_reg_classifiers.pkl',
#   'larslasso_classifiers.pkl'
# ]

# def load_all_classifiers():
#     clfs = []
#     for f in files_to_load:
#         fname = df_folder + '/' + f
#         print('Loaded', fname)
#         clf = joblib.load(fname)
#         clfs.append(clf)
#     return clfs

classifiers = [
  rf_clfs,
  gnb_clfs,
  bnb_clfs,
  mnb_clfs,
  dt_clfs,
  linear_reg_clfs,
  log_reg_clfs,
  log_reg_l1_clfs,
  ridge_reg_clfs,
  lasso_reg_clfs,
  larslasso_clfs
]

data = FragileFams()
# clf_dfs = load_all_classifiers()
clf_dfs = classifiers
fs_df = joblib.load(fs_file)
print('Classifier dfs: load complete.')

## ------------------------------------------------------------------------------------------------------ ##
##   STOPPAGE: code from here doesn't work, as it loads from our basic linear regression on continuous.
##   Need to work out how we ran linear regression across the continuous variables!!
## ------------------------------------------------------------------------------------------------------ ##


# lr_predictions = pd.read_csv('predictions/basic_lr.csv',
#                  names=['challengeID','gpa','grit','materialHardship','eviction','layoff','jobTraining'])
# cont_preds = lr_predictions[['challengeID','gpa', 'grit', 'materialHardship']]
# cont_preds.index = cont_preds.index + 1
# print('Continuous predictions loaded.')

""" Get the predictions for a classifier with given feature selection techniques for binary variables. """
# def get_predictions(clf_key,eviction='all',layoff='all',jobTraining='all',debug=False):
#     ev_fs = eviction
#     lay_fs = layoff
#     job_fs = jobTraining
#     clf_labels = ['rf', 'gnb', 'bnb', 'mnb', 'dt', 'linear_reg', 'log_reg_l1', 'log_reg_l2', 'ridge', 'lasso', 'lasso_lars']
#     _idx = clf_labels.index(clf_key)
#     clf_eviction = clf_dfs[_idx]['eviction'][ev_fs]
#     clf_layoff = clf_dfs[_idx]['layoff'][lay_fs]
#     clf_job = clf_dfs[_idx]['jobTraining'][job_fs]
#     _start = time.process_time()
#     eviction_preds = clf_eviction.predict_proba(data.full_set.filter(fs_df['eviction'][ev_fs]).values)[:,1]
#     layoff_preds = clf_layoff.predict_proba(data.full_set.filter(fs_df['layoff'][lay_fs]).values)[:,1]
#     job_preds = clf_job.predict_proba(data.full_set.filter(fs_df['jobTraining'][job_fs]).values)[:,1]
#     _end = time.process_time()
    
#     results = pd.DataFrame()
#     results['eviction'] = eviction_preds
#     results['layoff'] = layoff_preds
#     results['jobTraining'] = job_preds
#     if (debug):
#         print('Predictions complete in', _end - _start, 'seconds.')
#     results.index = results.index + 1
#     predictions = cont_preds.copy()
#     for col_name in results.columns:
#         predictions[col_name] = results[col_name].copy()
#     return predictions



