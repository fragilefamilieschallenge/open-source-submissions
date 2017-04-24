#==============================================================================
# COS 424
# Assignment 2: Fragile Families Challenge
# Authors:
#    Alex Rivas <amrivas@princeton.edu>
#    Ruoyang (Vincent) Peng <ruoyangp@princeton.edu>
#==============================================================================

import math
import numpy as np 
from sklearn import tree
from IPython.display import Image  
from sklearn.externals.six import StringIO
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas
import os
import sys
import tables
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV,LassoLarsIC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import brier_score_loss
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import resample

import time

get_ipython().magic(u'matplotlib inline')
#Setting Up Environment
workingdir = os.getcwd()
print(workingdir)



#%%
#Importing Chromosome #1 Data
train = pandas.read_csv('train.csv', sep=',', header=0)
prediction = pandas.read_csv('prediction.csv', sep=',', header=0)
backgroundfix = pandas.read_csv('backgroundfix.csv', sep=',', header=0)

#%% Preprocess data for each outcome
# Manually choose which outcome you are learning and forecasting

#GPAtrain = train[train.gpa.notnull()]
#GPAbackground = backgroundfix[backgroundfix.challengeID.isin(GPAtrain.challengeID)]

#GRITtrain = train[train.grit.notnull()]
#GRITbackground = backgroundfix[backgroundfix.challengeID.isin(GRITtrain.challengeID)]

#MHtrain = train[train.materialHardship.notnull()]
#MHbackground = backgroundfix[backgroundfix.challengeID.isin(MHtrain.challengeID)]

#EVtrain = train[train.eviction.notnull()]
#EVbackground = backgroundfix[backgroundfix.challengeID.isin(EVtrain.challengeID)]

#LOtrain = train[train.layoff.notnull()]
#LObackground = backgroundfix[backgroundfix.challengeID.isin(LOtrain.challengeID)]

JTtrain = train[train.jobTraining.notnull()]
JTbackground = backgroundfix[backgroundfix.challengeID.isin(JTtrain.challengeID)]


#%%
# Manually change the variable names according to the last cell
# record only survey questions with interger and floating responses
test=JTbackground.dtypes==np.int64
test2=JTbackground.dtypes==np.float64
test3=test | test2
X = JTbackground.loc[:,test3]
X = X.drop(['Unnamed: 0', 'idnum', 'challengeID'], 1) ## drop columns of IDs
X = X.values
Y = np.ones(JTtrain.shape[0])
for i, j in enumerate(JTtrain.ix[:, 4].astype('str')):
    if j=='True':
        Y[i] = 1
    if j=='False':
        Y[i] = 0
#%%
# Feature Selection (aka reducing dimensionality)
# In the report, we use sel2 only

# Removing features with zero variance
# removing features where XXX of the values are the same
p = 0.8
sel = VarianceThreshold(threshold = p*(1-p))
X_new=sel.fit_transform(X)
X_new.shape

# Univariate feature selection (set number of features = k)
sel2 = SelectKBest(mutual_info_classif, k=3000).fit(X, Y)
X_new2 = sel2.transform(X)
X_new2.shape
#%% Seperate train and test data
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
for train_index, test_index in sss.split(X_new2, Y):
   X_train, X_test = X_new2[train_index], X_new2[test_index]
   Y_train, Y_test = Y[train_index], Y[test_index]

#%%
## Logistic Regression with Lasso(CV)
Cs = np.logspace(-8, -2, 10)
scores = []
clf_l1_LR = LogisticRegression(penalty='l1', solver='liblinear')
for C in Cs:
    clf_l1_LR.C = C
    K = 3
    kf = KFold(n_splits=K)
    score = []
    for train, test in kf.split(X_train, Y_train):
        clf_l1_LR.fit(X_train[train], Y_train[train])
        y_pred = clf_l1_LR.predict_proba(X_train[test])
        score.append(brier_score_loss(Y_train[test], y_pred[:,1]))
    scores.append(np.mean(score))
C_optim_l1 = Cs[scores.index(min(scores))]
clf_l1_LR.C = C_optim_l1
clf_l1_LR.fit(X_train, Y_train)


## Visualize the average scores for different shrinkage parameters
plt.plot(np.log10(Cs), scores)

#%%
## Logistic Regression with Ridge (CV)
Cs = np.logspace(-4, 1, 6)
scores = []
clf_l2_LR = LogisticRegression(penalty='l2')
for C in Cs:
    clf_l2_LR.C = C
    K = 3
    kf = KFold(n_splits=K)
    score = []
    for train, test in kf.split(X_train, Y_train):
        clf_l2_LR.fit(X_train[train], Y_train[train])
        y_pred = clf_l2_LR.predict_proba(X_train[test])
        score.append(brier_score_loss(Y_train[test], y_pred[:,1]))
    scores.append(np.mean(score))
C_optim_l2 = Cs[scores.index(min(scores))]
clf_l2_LR.C = C_optim_l2
clf_l2_LR.fit(X_train, Y_train)

plt.plot(np.log10(Cs), scores)

#%% 
## Decision Tree Regressor
DT= DecisionTreeClassifier(max_depth=2)
DT.fit(X_train, Y_train)


#%%
## K Nearest Neighbours
neigh = KNeighborsClassifier(n_neighbors=20)
neigh.fit(X_train, Y_train) 
#%% 
## SVM
from sklearn.svm import SVC
clf_svc = SVC(probability=True)
clf_svc.fit(X_train, Y_train) 

#%%
## Evaluate model performance 
Y_pred_LR1 = clf_l1_LR.predict_proba(X_test)[:, 1]
Y_pred_LR2 = clf_l2_LR.predict_proba(X_test)[:, 1]
Y_pred_DT = DT.predict_proba(X_test)[:, 1]
Y_pred_KNN = neigh.predict_proba(X_test)[:, 1]
Y_pred_SVM = clf_svc.predict_proba(X_test)[:, 1]

mse_LR1 = mean_squared_error(Y_test, Y_pred_LR1)
mse_LR2 = mean_squared_error(Y_test, Y_pred_LR2)
mse_DT = mean_squared_error(Y_test, Y_pred_DT)
mse_KNN = mean_squared_error(Y_test, Y_pred_KNN)
mse_SVM = mean_squared_error(Y_test, Y_pred_SVM)

print('>LR1 MSE: ' + repr(mse_LR1))
print('>LR2 MSE: ' + repr(mse_LR2))
print('>DT MSE: ' + repr(mse_DT))
print('>KNN MSE: ' + repr(mse_KNN))
print('>SVM MSE: ' + repr(mse_SVM))

#%%
# Apply bootstrap to compute the confidence level of MSE
K_bs = 500 # number of samples in bootstrap
mse_bs = []
for k in range(K_bs):
    X_bs, Y_bs = resample(X_new2, Y)
    #model = LogisticRegression(C=C_optim_l1, penalty='l1', solver='liblinear')
    #model = DecisionTreeClassifier(max_depth=2)
    #model =  LogisticRegression(C=C_optim_l2, penalty='l2')
    model = KNeighborsClassifier(n_neighbors=20)
    #model = = DecisionTreeClassifier(max_depth=2)
    #model = clf_svc = SVC(probability=True)
    model.fit(X_bs, Y_bs)
    Y_bs_pred = model.predict_proba(X_new2)[:, 1]
    mse_bs.append(brier_score_loss(Y, Y_bs_pred))

plt.hist(mse_bs)
np.percentile(np.array(mse_bs), 2.5)
np.percentile(np.array(mse_bs), 97.5)
#%% Apply the model to the prediction.csv file
X_prediction = backgroundfix[backgroundfix.challengeID.isin(prediction.challengeID)]
X_prediction = X_prediction.loc[:,test3]
X_prediction = X_prediction.drop(['Unnamed: 0', 'idnum', 'challengeID'], 1)
X_prediction = X_prediction.values

X_prediction2 = sel2.transform(X_prediction)

#%%
# Manually choose which outcome you are learning and forecasting
#EVprediction = clf_l1_LR.predict_proba(X_prediction2)[:, 1]
#np.savetxt('EVprediction.csv', EVprediction, delimiter=',')

#LOprediction = clf_l1_LR.predict_proba(X_prediction2)[:, 1]
#np.savetxt('LOprediction.csv', LOprediction, delimiter=',')

JTprediction = clf_l1_LR.predict_proba(X_prediction2)[:, 1]
np.savetxt('JTprediction.csv', JTprediction, delimiter=',')