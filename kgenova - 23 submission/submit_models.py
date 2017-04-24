import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import model_selection
from a2core import *
from a2models import *
import os

refit_to_full_train = True
adapter_list = [MultiTaskRegressionAdapter]

def load_if_present_else_train(clf,X,y,clf_name,X_name):
    if any([isinstance(clf,t) for t in adapter_list]):
        if refit_to_full_train:
            print "Need to implement multioutput training on full set."
            assert(False)
        else:
            X_name = 'X_treg_train'
        fname = clf.clf.name() + '-' + X_name + '.pk'
        if os.path.isfile(fname):
            clf.clf = joblib.load(fname)
            return clf
        else:
            print "Warning: Need to fix code to send X_treg to train in submit."
            assert(False)
            clf.clf.fit(X,y)
            joblib.dump(clf.clf,fname)
            return clf
    fname = clf_name + '-' + X_name + '.pk'
    if os.path.isfile(fname):
        return joblib.load(fname)
    else:
        clf.fit(X,y)
        joblib.dump(clf,fname)
        return clf


# Set up models to compare
#multi_regressor = MultiClassElasticnetCVRFitter()
# gpa_regressor = MultiTaskRegressionAdapter(MultiClassElasticnetCVRFitter(),0)
# grit_regressor = MultiTaskRegressionAdapter(MultiClassElasticnetCVRFitter(),1)
# mhship_regressor = MultiTaskRegressionAdapter(MultiClassElasticnetCVRFitter(),2)
gpa_regressor = LinearRegressionFitter()
grit_regressor = LinearRegressionFitter()
mhship_regressor = LinearRegressionFitter()
#mhship_regressor = BayesianRidgeRegressionFitter()
# eviction_classifier = SGDClassifierFitter()#LogisticRegressionFitter()#EnsembleNBFitter()#NonlinearSVCGridSearchFitter()
# layoff_classifier = SGDClassifierFitter()#LogisticRegressionFitter()#EnsembleNBFitter()#DecisionTreeClassifierFitter()
# jobtraining_classifier = SGDClassifierFitter()#LogisticRegressionFitter()#EnsembleNBFitter()#DecisionTreeClassifierFitter()
eviction_classifier = LinearRegressionFitter()
layoff_classifier = LinearRegressionFitter()
jobtraining_classifier = LinearRegressionFitter()
clf_mode = 'Regression'
# eviction_classifier = SGDClassifierCVFitter()
# layoff_classifier = SGDClassifierCVFitter()
# jobtraining_classifier = SGDClassifierCVFitter()

fitters = [gpa_regressor,grit_regressor,mhship_regressor,eviction_classifier,layoff_classifier,jobtraining_classifier]
#fitters = [eviction_classifier,layoff_classifier,jobtraining_classifier]
# Load data
y_names = ['gpa','grit','materialHardship','eviction','layoff','jobTraining']
X_full = joblib.load('X.pk').todense()
X_full_name = 'X-full'

X_challengeIDs = joblib.load('X_challengeIDs.pk')

y_pred_all = [X_challengeIDs]

X_train_singles = [joblib.load('X_train_' + y_name + '.pk').todense() for y_name in y_names]

y_train_singles = [joblib.load('y_train_' + y_name + '.pk') for y_name in y_names]
for i in [0,1,2]:
    y_train_singles[i] = [float(v) for v in y_train_singles[i]]
if clf_mode == 'TrueFalse':
    for i in [3,4,5]:
        y_train_singles[i] = [int(v) for v in y_train_singles[i]]
else:
    for i in [3,4,5]:
        y_train_singles[i] = [float(v) for v in y_train_singles[i]]

if refit_to_full_train:
    X_train_single_names = ['X-trainval-'+y_name for y_name in y_names]
else:
    X_val_singles = []
    y_val_singles = []
    for i in xrange(len(X_train_singles)):
        X_train,X_val,y_train,y_val = model_selection.train_test_split(X_train_singles[i],y_train_singles[i], test_size=0.2,random_state=1337)
        X_train_singles[i] = X_train
        X_val_singles.append(X_val)
        y_train_singles[i] = y_train
        y_val_singles.append(y_val)
    X_train_single_names = ['X-single-'+y_name for y_name in y_names]

for i in xrange(6):
    y_name = y_names[i]
    print "Creating predictions for " + y_name
    fitter = fitters[i]
    model = load_if_present_else_train(fitter,X_train_singles[i],y_train_singles[i],fitter.name(),X_train_single_names[i])

    if i in [3,4,5] and model.name() == 'logistic-regression':
        X_tx = model.standardizer.transform(X_full)
        y_proba = model.clf.predict_proba(X_tx)[:,0]
        y_pred_all.append(y_proba)
        continue

    y_pred = model.predict(X_full)
    if i in [3,4,5] and clf_mode == 'TrueFalse':
        y_pred = decode_int_to_bool(y_pred)

    y_pred_all.append(y_pred)

print "Making submission..."
submission = [['challengeID',y_names[0],y_names[1],y_names[2],y_names[3],y_names[4],y_names[5]]]
submission.extend(transpose(y_pred_all))
subcsv = '\n'.join([','.join([str(x) for x in row]) for row in submission]) + '\n'
f = open('prediction.csv','w')
f.write(subcsv)
f.close()
