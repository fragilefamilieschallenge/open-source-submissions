#!/usr/bin/env python


import sys
import time
import argparse
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV

from scipy.stats import randint as sp_randint


NUM_JOBS = 4


# From ``MissingDataScript.py``, provided as a resource for Princeton cos424, Spring 2017
# def fillMissing(inputcsv, outputcsv):
#     # read input csv - takes time
#     df = pd.read_csv(inputcsv, low_memory=False)
#     # Fix date bug
#     df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
    
#     # replace NA's with mode
#     df = df.fillna(df.mode().iloc[0])
#     # if still NA, replace with 1
#     df = df.fillna(value=1)
#     # replace negative values with 1
#     num = df._get_numeric_data()
#     num[num < 0] = 1
#     # write filled outputcsv
#     num.to_csv(outputcsv, index=False)


def no_null_mask(labels, selector=None):
    if selector:
        good_row_mask = ~(labels.loc[:,selector].isnull())
    else:
        good_row_mask = ~(labels.iloc[:,1:].isnull().any(axis=1))

    return good_row_mask


#####################################################


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-bgpa", "--background_gpa", help="Path to data for training gpa", required=True)
    parser.add_argument("-bgrit", "--background_grit", help="Path to data for training grit", required=True)
    parser.add_argument("-bhard", "--background_hard", help="Path to data for training materialHardship", required=True)
    parser.add_argument("-t", "--train", help="Path to train.csv", required=True)
    parser.add_argument("-o", "--outfile", help="Path for output predictions.csv", required=True)
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")

    options = parser.parse_args()

    run(options)


def run(options):
    ofile = open(options.outfile, 'w')

    if options.verbose:
        print "Reading background files"
    gpa_data = pd.read_csv(options.background_gpa, low_memory=False)
    grit_data = pd.read_csv(options.background_grit, low_memory=False)
    hard_data = pd.read_csv(options.background_hard, low_memory=False)

    if options.verbose:
        print "Parsing and extracting non-null labels"
    train_labels = pd.read_csv(options.train)
    gpa_label_mask = no_null_mask(train_labels, selector="gpa")
    grit_label_mask = no_null_mask(train_labels, selector="grit")
    hard_label_mask = no_null_mask(train_labels, selector="materialHardship")

    if options.verbose:
        print "Selecting samples with enough data (non-null labels)"
    gpa_ids = train_labels.loc[gpa_label_mask,'challengeID'].values.flatten()
    grit_ids = train_labels.loc[grit_label_mask,'challengeID'].values.flatten()
    hard_ids = train_labels.loc[hard_label_mask,'challengeID'].values.flatten()

    # See http://stackoverflow.com/questions/12096252/use-a-list-of-values-to-select-rows-from-a-pandas-dataframe
    #     for information on isin and the pattern used here
    gpa_train_data = gpa_data[gpa_data['challengeID'].isin(gpa_ids)]
    grit_train_data = grit_data[grit_data['challengeID'].isin(grit_ids)]
    hard_train_data = hard_data[hard_data['challengeID'].isin(hard_ids)]
    
    if options.verbose:
        print "Ensuring ordering between samples and labels"
    gpas = dict(train_labels.loc[gpa_label_mask,['challengeID',"gpa"]].values)
    grits = dict(train_labels.loc[grit_label_mask,['challengeID',"grit"]].values)
    hards = dict(train_labels.loc[hard_label_mask,['challengeID',"materialHardship"]].values)

    ordered_gpas = []
    for cid in gpa_train_data.loc[:,'challengeID'].values:
        ordered_gpas.append(gpas[cid])

    ordered_grits = []
    for cid in grit_train_data.loc[:,'challengeID'].values:
        ordered_grits.append(grits[cid])

    ordered_hards = []
    for cid in hard_train_data.loc[:,'challengeID'].values:
        ordered_hards.append(hards[cid])

    # if options.verbose:
    #     print "Start training at: " + time.strftime("%H:%M:%S")
    # # See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # # and http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html
    # clf = linear_model.LogisticRegression(penalty='l1', C=0.8)
    # # clf.fit(X=good_data.iloc[:,:9001].values, y=gpa_labels.values.flatten())
    # clf.fit(X=good_data.iloc[:,:9001].values, y=gpa_labels)
    # if options.verbose:
    #     print "End training at:   " + time.strftime("%H:%M:%S")

    # Andrew: Do the chi2 feature selection
    # Note: This actually made things worse, so I left this at 1 for now, which means select all features
    chi2_fraction = 1
    max_features = 10000

    gpa_keep_indices = get_keep_indices(\
        gpa_train_data, ordered_gpas, gpa_ids, chi2_fraction, max_features)
    X_gpa = gpa_train_data[gpa_keep_indices].values
    y_gpa = ordered_gpas

    grit_keep_indices = get_keep_indices(\
        grit_train_data, ordered_grits, grit_ids, chi2_fraction, max_features)
    X_grit = grit_train_data[grit_keep_indices].values
    y_grit = ordered_grits

    hard_keep_indices = get_keep_indices(\
        hard_train_data, ordered_hards, hard_ids, chi2_fraction, max_features)
    X_hard = hard_train_data[hard_keep_indices].values
    y_hard = ordered_hards

    # if options.verbose:
    #     print "Scaling data"
    # See http://scikit-learn.org/stable/modules/preprocessing.html
    #     for information on preprocessing
    # scaler = preprocessing.StandardScaler().fit(X)
    # X_scaled = scaler.transform(X)


    # See http://scikit-learn.org/stable/modules/cross_validation.html
    #     for cross validation
    v = 0
    if options.verbose:
        print "Prep cross validation at: " + time.strftime("%H:%M:%S")
        v = 1
    # svm_reg = svm.SVR()
    # lasso_reg = linear_model.Lasso()
    # rf_reg = RandomForestRegressor()
    # reg = MLPRegressor() # Not fast


    # Hyperparameter tuning for random forest regressor
    rf_hyperparameter_space = {
        "n_estimators":      [50, 75, 100],
        "criterion":         ["mse"],
        "max_features":      ["auto", 0.75],
        "min_samples_split": [10, 25, 50]
    }

    # lasso_hyperparameter_space = {
    #     "alpha":         [1000, 10000],
    #     "fit_intercept": [True],
    #     "normalize":     [True, False],
    #     "tol":           [0.01, 0.1, 1.0],
    #     "positive":      [True, False],
    #     "selection":     ["random"],
    #     "max_iter":      [1000]
    # }

    # svm_hyperparameter_space = {
    #     "kernel":   ["rbf", "linear", "poly", "sigmoid"],
    #     "epsilon":  [0.01, 0.1, 1.0],
    #     "C":        [0.1, 1.0, 10],
    #     "degree":   [1, 2, 3, 4],
    #     "max_iter": [5000]
    # }

    # num_iterations = 32



    ####################################################################################################################################################
    # See http://scikit-learn.org/stable/modules/grid_search.html                                                                                      #
    #     for information on grid search for hpyerparameter tuning                                                                                     #
    # And http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV  #
    #     for information on randomized search                                                                                                         #
    # See http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html                                                          #
    #     for example usage (adapted below)                                                                                                            #
    ####################################################################################################################################################
    # search = GridSearchCV(rf_reg, param_grid=rf_hyperparameter_space, cv=3, scoring=make_scorer(r2_score, greater_is_better=True), verbose=v, refit=True, n_jobs=NUM_JOBS)
    # search = GridSearchCV(lasso_reg, param_grid=lasso_hyperparameter_space, cv=3, scoring=make_scorer(r2_score, greater_is_better=True), verbose=v, refit=True, n_jobs=NUM_JOBS)
    # search = RandomizedSearchCV(svm_reg, param_distributions=svm_hyperparameter_space, cv=3, scoring=make_scorer(r2_score, greater_is_better=True), verbose=v, refit=True, n_jobs=NUM_JOBS, n_iter=num_iterations)


    # if options.verbose:
    #     print "Start cross validation at: " + time.strftime("%H:%M:%S")
    # search.fit(X, y)
    # search.fit(X_scaled, y) # Lasso seems to do worse with scaled data

    # if options.verbose:
    #     print "End cross validation at:   " + time.strftime("%H:%M:%S")

    gpa_predict_X = gpa_data[gpa_keep_indices].values
    grit_predict_X = grit_data[grit_keep_indices].values
    hard_predict_X = hard_data[hard_keep_indices].values
    gpa_cids = list(gpa_data.loc[:,'challengeID'].values)
    grit_cids = list(grit_data.loc[:,'challengeID'].values)
    hard_cids = list(hard_data.loc[:,'challengeID'].values)

    # Parameters tuned through GridSearchCV
    if options.verbose:
        print "Training for gpa"
    # gpa_model = RandomForestRegressor(max_features="auto", min_samples_split=10, criterion="mse", n_estimators=75)
    gpa_model = GridSearchCV(RandomForestRegressor(), param_grid=rf_hyperparameter_space, cv=3, scoring=make_scorer(r2_score, greater_is_better=True), verbose=v, refit=True, n_jobs=NUM_JOBS)
    gpa_model.fit(X_gpa, y_gpa)
    gpas = {}

    if options.verbose:
        print "Training for grit"
    # grit_model = RandomForestRegressor(max_features="auto", min_samples_split=25, criterion="mse", n_estimators=50)
    grit_model = GridSearchCV(RandomForestRegressor(), param_grid=rf_hyperparameter_space, cv=3, scoring=make_scorer(r2_score, greater_is_better=True), verbose=v, refit=True, n_jobs=NUM_JOBS)
    grit_model.fit(X_grit, y_grit)
    grits = {}

    if options.verbose:
        print "Training for materialHardship"
    # hard_model = RandomForestRegressor(max_features="auto", min_samples_split=10, criterion="mse", n_estimators=50)
    hard_model = GridSearchCV(RandomForestRegressor(), param_grid=rf_hyperparameter_space, cv=3, scoring=make_scorer(r2_score, greater_is_better=True), verbose=v, refit=True, n_jobs=NUM_JOBS)
    hard_model.fit(X_hard, y_hard)
    hards = {}


    if options.verbose:
        print "\n"
        print "################################################################"
        print "\n"
        print "gpa tuning results"
        print "------------------"
        print "Best Score: " + str(gpa_model.best_score_)
        print "Best Parameters"
        print gpa_model.best_params_
        print "\n"
        print "grit tuning results"
        print "------------------"
        print "Best Score: " + str(grit_model.best_score_)
        print "Best Parameters"
        print grit_model.best_params_
        print "\n"
        print "materialHardship tuning results"
        print "------------------"
        print "Best Score: " + str(hard_model.best_score_)
        print "Best Parameters"
        print hard_model.best_params_
        print "\n"
        print "################################################################"
        print "\n"

    cids = sorted(list(set(gpa_cids).intersection(set(grit_cids)).intersection(set(hard_cids))))

    if options.verbose:
        print "Predicting rest of labels"
    gpa_predicts = gpa_model.predict(gpa_predict_X)
    for i in xrange(len(cids)):
        gpas[cids[i]] = gpa_predicts[i]

    grit_predicts = grit_model.predict(grit_predict_X)
    for i in xrange(len(cids)):
        grits[cids[i]] = grit_predicts[i]

    hard_predicts = hard_model.predict(hard_predict_X)
    for i in xrange(len(cids)):
        hards[cids[i]] = hard_predicts[i]

    # Output
    if options.verbose:
        print "Writing predictions to " + options.outfile
    dumbpredict = False

    ofile.write('"challengeID","gpa","grit","materialHardship","eviction","layoff","jobTraining"\n')

    for cid in cids:
        ofile.write(str(cid))
        ofile.write(",")
        ofile.write(str(gpas[cid]))
        ofile.write(",")
        ofile.write(str(grits[cid]))
        ofile.write(",")
        ofile.write(str(hards[cid]))
        ofile.write(",")
        ofile.write(str(dumbpredict))
        ofile.write(",")
        ofile.write(str(dumbpredict))
        ofile.write(",")
        ofile.write(str(dumbpredict))
        ofile.write("\n")
        ofile.flush()

    ofile.close()


def get_keep_indices(background_df, train_labels, challenge_ids, fraction = 1, max_features = 10000):
    '''
    :return indices corresponding to features to keep
    '''
    k = min(int(background_df.shape[1] * fraction), max_features)
    select = SelectKBest(lambda X, y: np.array([-1 * t for t in chi2(X, y)[0].tolist()]), k)
    train_labels = [round(x) for x in train_labels]
    select.fit(background_df.values, train_labels)
    keep_indices = np.where(select.get_support())[0].tolist()
    drop_indices = list(set(range(background_df.shape[1])) - set(keep_indices))
    print "  - Using this many features: %s" % len(keep_indices)
    if len(drop_indices) < 100:
        print "    - Dropping these columns: %s" % background_df.columns[drop_indices].values.tolist()
    if len(keep_indices) < 100:
        print "    - Keeping these columns: %s" % background_df.columns[keep_indices].values.tolist()
    return keep_indices


if __name__ == "__main__":
    main()

