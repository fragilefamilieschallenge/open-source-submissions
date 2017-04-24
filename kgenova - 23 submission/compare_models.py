import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import model_selection
from a2core import *
from a2models import *
import os

def load_if_present_else_train(clf,X,y,clf_name,X_name):
    fname = clf_name + '-' + X_name + '.pk'
    if (os.path.isfile(fname)):
        return joblib.load(fname)
    else:
        clf.fit(X,y)
        joblib.dump(clf,fname)
        return clf

# Set up models to compare
single_regressors = [BayesianRidgeRegressionFitter(),ElasticNetCVRegressionFitter()]
single_classifiers = [NonlinearSVCGridSearchFitter(),DecisionTreeClassifierFitter(),EnsembleNBFitter(),EnsembleNBCVFitter(),LogisticRegressionFitter(),SGDClassifierFitter()]#,GaussianProcessClassifierFitter()]
multiple_regressors = [MultiClassElasticnetCVRFitter()]
# Load data
y_names = ['gpa','grit','materialHardship','eviction','layoff','jobTraining']
X_full = joblib.load('X.pk').todense()
X_full_name = 'X-full'

X_train_singles = [joblib.load('X_train_' + y_name + '.pk').todense() for y_name in y_names]

y_train_singles = [joblib.load('y_train_' + y_name + '.pk') for y_name in y_names]
for i in [0,1,2]:
    y_train_singles[i] = [float(v) for v in y_train_singles[i]]
for i in [3,4,5]:
    y_train_singles[i] = [int(v) for v in y_train_singles[i]]

X_val_singles = []
y_val_singles = []
for i in xrange(len(X_train_singles)):
    X_train,X_val,y_train,y_val = model_selection.train_test_split(X_train_singles[i],y_train_singles[i], test_size=0.2,random_state=1337)
    X_train_singles[i] = X_train
    X_val_singles.append(X_val)
    y_train_singles[i] = y_train
    y_val_singles.append(y_val)
X_train_single_names = ['X-single-'+y_name for y_name in y_names]

X_treg = joblib.load('X_treg.pk').todense()
y_treg = joblib.load('y_treg.pk')
X_tclf = joblib.load('X_tclf.pk').todense()
y_tclf = joblib.load('y_tclf.pk')

X_treg_train,X_treg_val,y_treg_train,y_treg_val = model_selection.train_test_split(X_treg,y_treg, test_size=0.2,random_state=1337)
X_tclf_train,X_tclf_val,y_tclf_train,y_tclf_val = model_selection.train_test_split(X_tclf,y_tclf, test_size=0.2,random_state=1337)


for i in [0,1,2]:
    y_name = y_names[i]
    print "Comparing single regressors on " + y_name
    for regressor in single_regressors:
        print "Evaluating regressor " + regressor.name()
        regressor = load_if_present_else_train(regressor,X_train_singles[i],y_train_singles[i],regressor.name(),X_train_single_names[i])
        print "Performance on training set: "
        y_pred = regressor.predict(X_train_singles[i])
        print "MSE: " + str(metrics.mean_squared_error(y_train_singles[i],y_pred))
        print "Explained variance: " + str(metrics.explained_variance_score(y_train_singles[i],y_pred))
        print "R2: " + str(metrics.r2_score(y_train_singles[i],y_pred))
        print ""
        print "Performance on validation set: "
        y_pred = regressor.predict(X_val_singles[i])
        print "MSE: " + str(metrics.mean_squared_error(y_val_singles[i],y_pred))
        print "Explained variance: " + str(metrics.explained_variance_score(y_val_singles[i],y_pred))
        print "R2: " + str(metrics.r2_score(y_val_singles[i],y_pred))
        print "\n"
    print "\n\n"

for i in [3,4,5]:
    y_name = y_names[i]
    print "Comparing single classifiers on " + y_name
    for classifier in single_classifiers:
        print "Evaluating classifier " + classifier.name()
        classifier = load_if_present_else_train(classifier,X_train_singles[i],y_train_singles[i],classifier.name(),X_train_single_names[i])
        print "Performance on training set: "
        y_pred = classifier.predict(X_train_singles[i])
        print "Accuracy: " + str(metrics.accuracy_score(y_train_singles[i],y_pred))
        print "F1 Score: " + str(metrics.f1_score(y_train_singles[i],y_pred))
        print "PrecRecFscoreSupport: " + str(metrics.precision_recall_fscore_support(y_train_singles[i],y_pred))
        y_score = classifier.decision(X_train_singles[i])
        if type(y_score) == list:
            print "ROC_AUC: " + str(metrics.roc_auc_score(y_train_singles[i],y_score))
        print ""
        print "Performance on validation set: "
        y_pred = classifier.predict(X_val_singles[i])
        print "Accuracy: " + str(metrics.accuracy_score(y_val_singles[i],y_pred))
        print "F1 Score: " + str(metrics.f1_score(y_val_singles[i],y_pred))
        print "PrecRecFscoreSupport: " + str(metrics.precision_recall_fscore_support(y_val_singles[i],y_pred))
        y_score = classifier.decision(X_val_singles[i])
        if type(y_score) == list:
            print "ROC_AUC: " + str(metrics.roc_auc_score(y_val_singles[i],y_score))
        print "\n"
    print "\n\n"

# Multiple Regression:
print "Comparing multiple regressors..."
for regressor in multiple_regressors:
    print "Evaluating regressor " + regressor.name()
    regressor = load_if_present_else_train(regressor,X_treg_train,y_treg_train,regressor.name(),'X_treg_train')

    y_pred_train = transpose(regressor.predict(X_treg_train))
    y_pred_val = transpose(regressor.predict(X_treg_val))
    y_truth_train = transpose(y_treg_train)
    y_truth_val = transpose(y_treg_val)
    for i in [0,1,2]: #NOTE For multi-clfs- need to index y by [0,1,2] and results by [3,4,5]
        print "Evaluating multiclassifier on dimension " + y_names[i]
        print "Performance on training set: "
        print "MSE: " + str(metrics.mean_squared_error(y_truth_train[i],y_pred_train[i]))
        print "Explained variance: " + str(metrics.explained_variance_score(y_truth_train[i],y_pred_train[i]))
        print "R2: " + str(metrics.r2_score(y_truth_train[i],y_pred_train[i]))
        print ""

        print "Performance on validation set: "
        print "MSE: " + str(metrics.mean_squared_error(y_truth_val[i],y_pred_val[i]))
        print "Explained variance: " + str(metrics.explained_variance_score(y_truth_val[i],y_pred_val[i]))
        print "R2: " + str(metrics.r2_score(y_truth_val[i],y_pred_val[i]))
        print "\n"
