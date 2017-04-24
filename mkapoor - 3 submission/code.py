#!/usr/local/bin/python
# Mihika Kapoor
# COS 424 Assignment 2

import numpy as np
import pandas as pd
import time
import csv
import matplotlib.pyplot as plt
import random
import re
import sys

from numpy import genfromtxt
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def splitbackground(backgrounddf):
	
	traindf = pd.read_csv('trainoutput.csv', low_memory=False)
	tempdf = pd.merge(backgrounddf, traindf, how='inner', on=['challengeID'])

	temp2df = pd.merge(backgrounddf, traindf, how='outer', on=['challengeID'])

	temp3df = temp2df.merge(tempdf,on=['challengeID'])
	temp4df = temp2df[~temp2df.challengeID.isin(temp3df.challengeID)]


	tempdf = tempdf.drop('gpa', 1)
	tempdf = tempdf.drop('grit', 1)
	tempdf = tempdf.drop('materialHardship', 1)
	tempdf = tempdf.drop('eviction', 1)
	tempdf = tempdf.drop('layoff', 1)
	tempdf = tempdf.drop('jobTraining', 1)
	tempdf.to_csv('matchedbackground.csv', index=False)

	temp4df = temp4df.drop('gpa', 1)
	temp4df = temp4df.drop('grit', 1)
	temp4df = temp4df.drop('materialHardship', 1)
	temp4df = temp4df.drop('eviction', 1)
	temp4df = temp4df.drop('layoff', 1)
	temp4df = temp4df.drop('jobTraining', 1)
	temp4df.to_csv('unmatchedbackground.csv', index=False)

# provided
def fillMissing(inputcsv, outputcsv):
    
    # read input csv - takes time
    df = pd.read_csv(inputcsv, low_memory=False)
    # Fix date bug
    # df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
    
    # replace NA's with mode
    df = df.fillna(df.mode().iloc[0])
    # if still NA, replace with 1
    df = df.fillna(value=1)
    # replace negative values with 1
    num = df._get_numeric_data()
    num[num < 0] = 1

    df = df.astype(int)
    # write filled outputcsv
    df.to_csv(outputcsv, index=False)

def excludeMissing(inputcsv, outputcsv):
    
    # read input csv - takes time
    df = pd.read_csv(inputcsv, low_memory=False)
    
    # remove remaining strings
    df = df.select_dtypes(exclude=['object'])

    # write filled outputcsv
    df.to_csv(outputcsv, index=False)

def predictFitBinary(clf, train_data, train_class, test_data, test_class):
	clf.fit(train_data, train_class)
	predicted = clf.predict(test_data)
	print "score: ", metrics.accuracy_score(test_class, predicted)
	print "f1 score: ", metrics.f1_score(test_class, predicted)
	print metrics.classification_report(test_class, predicted)

def rmseContinuous(clf, train_data, train_class, test_data, test_class):
	clf.fit(train_data, train_class)
	predicted = clf.predict(test_data)
	print "RMSE: ", mean_squared_error(test_class, predicted)
	print "R^2: ", r2_score(test_class, predicted)


def binaryClassifiers(train_bag, train_class, test_bag, test_class):
	print "Naive Bayes"
	start = time.time()
	naive = MultinomialNB()
	predictFitBinary(naive, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print "time: ", (end - start)
	# calcROC(naive, test_bag, test_class, 'NB')
	print

	print "Logistic Regression"
	start = time.time()
	logreg = linear_model.LogisticRegression()
	predictFitBinary(logreg, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print "time: ", (end - start)
	# calcROC(logreg, test_bag, test_class, 'LR' )
	print

	print "SVM (Linear Kernel)"
	svml = SVC(kernel='linear', probability=True)
	start = time.time()
	predictFitBinary(svml, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print "time: ", (end - start)
	# calcROC(svml, test_bag, test_class, 'SVML')
	print

	print "SVM (Gaussian Kernel)"
	start = time.time()
	svmg = SVC(kernel='rbf', probability=True)
	predictFitBinary(svmg, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print "time: ", (end - start)
	# calcROC(svmg, test_bag, test_class, 'SVMG')
	print

	print "Decision Tree"
	start = time.time()
	dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
	predictFitBinary(dt, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print "time: ", (end - start)
	# calcROC(dt, test_bag, test_class, 'DT')

	print "K Nearest Neighbors"
	start = time.time()
	knn = linear_model.LogisticRegression()
	predictFitBinary(logreg, train_bag, train_class, test_bag, test_class)
	end = time.time()
	print "time: ", (end - start)
	# calcROC(logreg, test_bag, test_class, 'LR' )
	print

	# adapted from http://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
	print "Random Forest Classifier"
	seed = 7
	num_trees = 100
	max_features = 100
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
	predictFitBinary(rf, train_bag, train_class, test_bag, test_class)
	results = model_selection.cross_val_score(rf, train_bag, train_class, cv=kfold)
	print(results.mean())

# see http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html
def continuousClassifiers(train_bag, train_class, test_bag, test_class):
	print "Random Forest"
	rf = RandomForestRegressor()
	rmseContinuous(rf, train_bag, train_class, test_bag, test_class)

	alpha = 0.1
	print "Elastic Net"
	enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
	rmseContinuous(enet, train_bag, train_class, test_bag, test_class)

	print "Lasso"
	lasso = Lasso(alpha=alpha)
	rmseContinuous(lasso, train_bag, train_class, test_bag, test_class)

	print "Ridge"
	ridge = Ridge(alpha=alpha,normalize=True)
	rmseContinuous(ridge, train_bag, train_class, test_bag, test_class)




def tocsv():
	df = pd.read_csv('matchedbackground.csv', low_memory=False)
	df_labels = pd.read_csv('trainoutput.csv', low_memory=False)
	X = df
	test_data = pd.read_csv('new_background.csv', low_memory=False)
	
	y = df_labels.pop('gpa')
	rf = RandomForestRegressor()
	rf.fit(X, y)
	df1 = pd.DataFrame(rf.predict(test_data))

	y2 = df_labels.pop('grit')
	rf2 = RandomForestRegressor()
	rf2.fit(X, y2)
	df2 = pd.DataFrame(rf2.predict(test_data))

	y3 = df_labels.pop('materialHardship')
	rf3 = RandomForestRegressor()
	rf3.fit(X, y3)
	df3 = pd.DataFrame(rf3.predict(test_data))

	seed = 7
	num_trees = 100
	max_features = 100
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	
	rfnew = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
	y4 = df_labels.pop('eviction')
	rfnew.fit(X, y4)
	df4 = pd.DataFrame(rfnew.predict(test_data))

	rfnew2 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
	y5 = df_labels.pop('layoff')
	rfnew2.fit(X, y5)
	df5 = pd.DataFrame(rfnew2.predict(test_data))

	rfnew3 = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
	y6 = df_labels.pop('jobTraining')
	rfnew3.fit(X, y6)
	df6 = pd.DataFrame(rfnew3.predict(test_data))


	dfchallenge = pd.DataFrame(test_data.pop('challengeID'))
	dffinal = pd.concat([dfchallenge, df1, df2, df3, df4, df5, df6], axis=1)
	dffinal.to_csv('prediction.csv', index=False)

	
	

def main():
	fillMissing('train.csv', 'trainoutput.csv')
	filleddf = pd.read_csv('trainoutput.csv', low_memory=False)
	excludeMissing('output.csv', 'new_output.csv')
	excludeeddf = pd.read_csv('output.csv', low_memory=False)

	fillMissing('background.csv', 'new_background.csv')
	filleddf = pd.read_csv('new_background.csv', low_memory=False)
	excludeMissing('output.csv', 'new_output.csv')
	excludeeddf = pd.read_csv('output.csv', low_memory=False)

	backgrounddf = pd.read_csv('new_background.csv', low_memory=False)
	splitbackground(backgrounddf)

	df = pd.read_csv('matchedbackground.csv', low_memory=False)
	df_labels = pd.read_csv('trainoutput.csv', low_memory=False)
	y = df_labels.pop('eviction')
	X = df


	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
	# binaryClassifiers(X_train, y_train, X_test, y_test)
	binaryClassifiers(X_train, y_train, X_test, y_test)
	tocsv()

main()


