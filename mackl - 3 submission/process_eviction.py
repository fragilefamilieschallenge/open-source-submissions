import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import time
from statsmodels.stats import api as sms
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from sklearn.svm import SVC


import matplotlib.pyplot as plt
#from sklearn.cross_validation import Bootstrap

## NEWBACKGROUND.CSV GENERATED FROM R SCRIPT ##
df = pd.read_csv("newbackground.csv", low_memory=False)


df = df.sort_values(by = 'challengeID', ascending = 1);
#print(df.iloc[[5]])
# num = df._get_numeric_data()
# num[num < 0] = 1

## temporarily change all NA to -99
df = df.fillna(value = np.nan)
#print(df.hv5_ppvtae)
dfcopy = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

#print(dfcopy)
#print(dfcopy.hv5_ppvtae)
#print(df.ffcc_famsurvey_f2)


## get columns that only have strings ##
#print(df)
# nonints = list(dfcopy.select_dtypes(exclude=['int64', 'float64']).columns)
# print(dfcopy.hv5_ppvtae)
#print(dfcopy.hv5_ppvtpr)
#print(nonints)

#print(np.nansum(dfcopy.hv5_ppvtae.values))
#print(np.nansum(dfcopy.hv5_ppvtpr))

categorical = list()
for column in dfcopy:
# # 	print(df[column])
	if dfcopy[column].isnull().sum() == len(dfcopy[column]):
		categorical.append(column)

#print(categorical)
for c in categorical:
	le = LabelEncoder()
	df[c] = le.fit_transform(df[c])


df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

start_time = time.time()
imputer = preprocessing.Imputer(strategy = "most_frequent")
df = imputer.fit_transform(df) 
impute_time = (time.time() - start_time)
print("Imputation time")
print(impute_time)
print("Number of features")
print(df.shape)
df = pd.DataFrame(df, columns = dfcopy.columns)

# PRINT MOST IMPORTANT FEATURES
# top10 = [0, 6, 16,  22,  38,  69,  81,  88, 103, 106, 129, 130, 131, 132, 133, 134, 136, 137,138, 168, 169, 170, 172 ,174, 188]
# print(df.columns[top10])
# print(df.m1a15)
outcomes = pd.read_csv("train.csv", low_memory=False)
outcomes_eviction = outcomes[pd.notnull(outcomes['eviction'])]


# ##################### FOR PREDICTIONS ############################
# evictions = [None]*4243
# print("CONDUCTING PREDICTION")

# for i in range(0, len(outcomes_eviction)):
# 	index = outcomes_eviction.challengeID.values[i]
# 	evictions[index] = outcomes_eviction.eviction.values[i]
# test_ids = list()
# for i in range(0, 4243):
# 	if (i not in outcomes_eviction.challengeID.values):
# 		test_ids.append(i)
# print(test_ids)
# test_ids.pop(0)
# test_data = df.loc[df['challengeID'].isin(test_ids)]
# train_data =  df.loc[df['challengeID'].isin(outcomes_eviction.challengeID)]
# clf = RandomForestClassifier(n_estimators = 100,criterion = "gini", max_depth = 10)
# start_time = time.time()
# test_outcomes = [int(i) for i in outcomes_eviction.eviction]
# clf.fit(train_data, test_outcomes)
# rfpred_test = clf.predict(test_data)

# print(len(test_outcomes))
# print("LENGTHS")
# print(len(test_ids))
# print(len(rfpred_test))

# for i in range(0, len(test_ids)):
# 	evictions[test_ids[i]] = rfpred_test[i]

# evictions.pop(0)
# for i in range(0, len(evictions)):
# 	if (evictions[i] == 0):
# 		evictions[i] = False
# 	else:
# 		evictions[i] = True
# print(evictions)

# rows = zip(evictions)
# import csv
# with open('predict_eviction.csv', 'wb') as f:
# 	writer = csv.writer(f, delimiter = ',')
# 	for row in rows:
# 		writer.writerow(row)

# ##################################################################





#################### FOR TRAINING ##################

# Get rows of training data that have reported eviction (exclude those with NA)


df = df.loc[df['challengeID'].isin(outcomes_eviction.challengeID)]

# df.to_csv('numbackground.csv', index=False)






# selector = VarianceThreshold()
# df = selector.fit_transform(df)
# print("Variance Threshold")
# print(df.shape)

target = [int(i) for i in outcomes_eviction.eviction]






### Begin Classification Process ###

from sklearn.model_selection import StratifiedKFold
k = 5
skf = StratifiedKFold(n_splits = k, shuffle=True)
count = 0


## initialize accumulators and averages ##
print("Starting to train model...")
nb_accuracy = 0.0
nb_precision = 0
nb_recall = 0
nb_f1 = 0
nb_precision_neg = 0
nb_recall_neg = 0
nb_f1_neg = 0
nb_time = 0

lr_accuracy = 0.0
lr_precision = 0
lr_recall = 0
lr_f1 = 0
lr_precision_neg = 0
lr_recall_neg = 0
lr_f1_neg = 0
lr_time = 0
lr_means = list()

knn_accuracy = 0.0
knn_precision = 0
knn_recall = 0
knn_f1 = 0
knn_precision_neg = 0
knn_recall_neg = 0
knn_f1_neg = 0
knn_time = 0
knn_means = list()

rf_accuracy = 0.0
rf_precision = 0
rf_recall = 0
rf_f1 = 0
rf_precision_neg = 0
rf_recall_neg = 0
rf_f1_neg = 0
rf_time = 0
rf_means = list()

dt_accuracy = 0.0
dt_precision = 0
dt_recall = 0
dt_f1 = 0
dt_precision_neg = 0
dt_recall_neg = 0
dt_f1_neg = 0
dt_time = 0

gp_accuracy = 0.0
gp_precision = 0
gp_recall = 0
gp_f1 = 0
gp_precision_neg = 0
gp_recall_neg = 0
gp_f1_neg = 0
gp_time = 0
gp_means = list()

svm_accuracy = 0.0
svm_precision = 0
svm_recall = 0
svm_f1 = 0
svm_precision_neg = 0
svm_recall_neg = 0
svm_f1_neg = 0
svm_time = 0

# import csv
# with open("./test.csv", "wb") as f:
# 	writer = csv.writer(f) 
# 	writer.writerows(df)

for index_train, index_test in skf.split(df, target):
	#print(index_train)
	#print(index_test)
	X_train = df.values[index_train,:]
	X_test = df.values[index_test,:]
	y_train = [target[i] for i in index_train]
	y_test = [target[i] for i in index_test]
	print(X_train.shape)
 # FEATURE SELECTION ##

	# selector = SelectKBest(chi2, k=10).fit(X_train, y_train)
	# #lasso = LinearSVC(C=0.01, penalty="l1", dual=False)
	# # # lasso = GradientBoostingClassifier()
	# # selector = SelectFromModel(lasso)
	# # selector.fit(X_train, y_train)
		
	# # selector = SelectPercentile(chi2, percentile = 70).fit(X_train, y_train)
	# idxs = selector.get_support(indices = True)
	# print(idxs)

	# X_train= selector.transform(X_train)
	# X_test = X_test[:,idxs]
	# print(X_train.shape)

	# from sklearn import preprocessing
	# scaler = preprocessing.StandardScaler()
	# X_train = scaler.fit_transform(X_train)
	# X_test = scaler.fit_transform(X_test)



	# print("******** Naive Bayes ******")
	# clf = MultinomialNB()
	# clf.fit(X_train, y_train)
	# nbpred_test = clf.predict(X_test)
	# start_time = time.time()
	# nb_time += (1.0/k)* (time.time() - start_time)
	# nb_accuracy += (1.0/k)*np.mean(nbpred_test == y_test)
	# nb_precision += (1.0/k)*precision_score(y_test, nbpred_test, average = "binary", pos_label = 0)
	# nb_recall += (1.0/k)*recall_score(y_test, nbpred_test, average = "binary", pos_label = 0)
	# nb_f1 += (1.0/k)*f1_score(y_test, nbpred_test, average = "binary", pos_label = 0)
	# nb_precision_neg += (1.0/k)*precision_score(y_test, nbpred_test, average = "binary", pos_label = 0)
	# nb_recall_neg += (1.0/k)*recall_score(y_test, nbpred_test, average = "binary",pos_label = 0)
	# nb_f1_neg += (1.0/k)*f1_score(y_test, nbpred_test, average = "binary", pos_label = 0)
	# print("**********************************")


	# if (count == 1):
	# 	nbpred_prob = clf.predict_proba(X_test)
	# 	fpr1, tpr1, thresholds1 = roc_curve(y_test, nbpred_prob[:,1])
	# 	roc_auc1 = auc(fpr1, tpr1)
	# 	plt.clf()
	# 	plt.plot(fpr1, tpr1, label = 'NB (area = %0.2f)' % roc_auc1)

	print("******** Logistic Regression ******")
	clf = LogisticRegression()
	start_time = time.time()
	clf.fit(X_train, y_train)
	lrpred_test = clf.predict(X_test)
	lr_time += (1.0/k)* (time.time() - start_time)
	lr_accuracy += (1.0/k)*np.mean(lrpred_test == y_test)
	lr_precision += (1.0/k)*precision_score(y_test, lrpred_test, average = "binary", pos_label = 0)
	lr_recall += (1.0/k)*recall_score(y_test, lrpred_test, average = "binary", pos_label = 0)
	lr_f1 += (1.0/k)*f1_score(y_test, lrpred_test, average = "binary", pos_label = 0)
	lr_precision_neg += (1.0/k)*precision_score(y_test, lrpred_test, average = "binary", pos_label = 0)
	lr_recall_neg += (1.0/k)*recall_score(y_test, lrpred_test, average = "binary",pos_label = 0)
	lr_f1_neg += (1.0/k)*f1_score(y_test, lrpred_test, average = "binary", pos_label = 0)
	lr_means.append(np.mean(lrpred_test == y_test))
	print("**********************************")

	if (count == 1):
		lrpred_prob = clf.predict_proba(X_test)
		fpr4, tpr4, thresholds4	 = roc_curve(y_test, lrpred_prob[:,1])
		roc_auc4 = auc(fpr4, tpr4)
		plt.plot(fpr4, tpr4, label = 'LR (area = %0.2f)' % roc_auc4)

	print("******** K-Nearest Neighbors ******")
	clf = KNeighborsClassifier(n_neighbors = 10, algorithm ='kd_tree')
	start_time = time.time()
	clf.fit(X_train, y_train)
	knnpred_test = clf.predict(X_test)
	knn_time += (1.0/k)* (time.time() - start_time)
	knn_accuracy += (1.0/k)*np.mean(knnpred_test == y_test)
	knn_precision += (1.0/k)*precision_score(y_test, knnpred_test, average = "binary", pos_label = 0)
	knn_recall += (1.0/k)*recall_score(y_test, knnpred_test, average = "binary", pos_label = 0)
	knn_f1 += (1.0/k)*f1_score(y_test, knnpred_test, average = "binary", pos_label = 0)
	knn_precision_neg += (1.0/k)*precision_score(y_test, knnpred_test, average = "binary", pos_label = 0)
	knn_recall_neg += (1.0/k)*recall_score(y_test, knnpred_test, average = "binary",pos_label = 0)
	knn_f1_neg += (1.0/k)*f1_score(y_test, knnpred_test, average = "binary", pos_label = 0)
	knn_means.append(np.mean(knnpred_test == y_test))
	print("**********************************")

	if (count == 1):
		knnpred_prob = clf.predict_proba(X_test)
		fpr2, tpr2, thresholds2 = roc_curve(y_test, knnpred_prob[:,1])
		roc_auc2 = auc(fpr2, tpr2)
		plt.plot(fpr2, tpr2, label = 'knn (area = %0.2f)' % roc_auc2)

	print("******** Random Forest ******")
	clf = RandomForestClassifier(n_estimators = 100,criterion = "gini", max_depth = 10)
	start_time = time.time()
	clf.fit(X_train, y_train)
	rfpred_test = clf.predict(X_test)
	rf_time += (1.0/k)* (time.time() - start_time)
	rf_accuracy += (1.0/k)*np.mean(rfpred_test == y_test)
	rf_precision += (1.0/k)*precision_score(y_test, rfpred_test, average = "binary", pos_label = 0)
	rf_recall += (1.0/k)*recall_score(y_test, rfpred_test, average = "binary", pos_label = 0)
	rf_f1 += (1.0/k)*f1_score(y_test, rfpred_test, average = "binary", pos_label = 0)
	rf_precision_neg += (1.0/k)*precision_score(y_test, rfpred_test, average = "binary", pos_label = 0)
	rf_recall_neg += (1.0/k)*recall_score(y_test, rfpred_test, average = "binary",pos_label = 0)
	rf_f1_neg += (1.0/k)*f1_score(y_test, rfpred_test, average = "binary", pos_label = 0)
	rf_means.append(np.mean(rfpred_test == y_test))
	print("**********************************")

	if (count == 1):
		rfpred_prob = clf.predict_proba(X_test)
		fpr5, tpr5, thresholds5	 = roc_curve(y_test, rfpred_prob[:,1])
		roc_auc5 = auc(fpr5, tpr5)
		plt.plot(fpr5, tpr5, label = 'RF (area = %0.2f)' % roc_auc5)

	print("******** Decision Tree ******")
	clf = DecisionTreeClassifier(criterion = "gini")
	start_time = time.time()
	clf.fit(X_train, y_train)
	dtpred_test = clf.predict(X_test)
	dt_time +=(1.0/k)*  (time.time() - start_time)
	dt_accuracy += (1.0/k)*np.mean(dtpred_test == y_test)
	dt_precision += (1.0/k)*precision_score(y_test, dtpred_test, average = "binary", pos_label = 0)
	dt_recall += (1.0/k)*recall_score(y_test, dtpred_test, average = "binary", pos_label = 0)
	dt_f1 += (1.0/k)*f1_score(y_test, dtpred_test, average = "binary", pos_label = 0)
	dt_precision_neg += (1.0/k)*precision_score(y_test, dtpred_test, average = "binary", pos_label = 0)
	dt_recall_neg += (1.0/k)*recall_score(y_test, dtpred_test, average = "binary",pos_label = 0)
	dt_f1_neg += (1.0/k)*f1_score(y_test, dtpred_test, average = "binary", pos_label = 0)
	print("**********************************")

	if (count == 1):
		treepred_prob = clf.predict_proba(X_test)
		fpr6, tpr6, thresholds6	 = roc_curve(y_test, treepred_prob[:,1])
		roc_auc6 = auc(fpr6, tpr6)
		plt.plot(fpr6, tpr6, label = 'DT (area = %0.2f)' % roc_auc6)


	# print("******** SVM ******")
	# clf = SVC(kernel = 'linear', probability = True)
	# start_time = time.time()
	# clf.fit(X_train, y_train)
	# svmpred_test = clf.predict(X_test)
	# svm_time += (time.time() - start_time)
	# svm_accuracy += (1.0/k)*np.mean(svmpred_test == y_test)
	# svm_precision += (1.0/k)*precision_score(y_test, svmpred_test, average = "binary", pos_label = 0)
	# svm_recall += (1.0/k)*recall_score(y_test, svmpred_test, average = "binary", pos_label = 0)
	# svm_f1 += (1.0/k)*f1_score(y_test, svmpred_test, average = "binary", pos_label = 0)
	# svm_precision_neg += (1.0/k)*precision_score(y_test, svmpred_test, average = "binary", pos_label = 0)
	# svm_recall_neg += (1.0/k)*recall_score(y_test, svmpred_test, average = "binary",pos_label = 0)
	# svm_f1_neg += (1.0/k)*f1_score(y_test, svmpred_test, average = "binary", pos_label = 0)
	print("**********************************")

	print("******** Gaussian Process ******")
	clf = GaussianProcessClassifier()
	start_time = time.time()
	clf.fit(X_train, y_train)
	gppred_test = clf.predict(X_test)
	gp_time += (1.0/k)* (time.time() - start_time)
	gp_accuracy += (1.0/k)*np.mean(gppred_test == y_test)
	gp_precision += (1.0/k)*precision_score(y_test, gppred_test, average = "binary", pos_label = 0)
	gp_recall += (1.0/k)*recall_score(y_test, gppred_test, average = "binary", pos_label = 0)
	gp_f1 += (1.0/k)*f1_score(y_test, gppred_test, average = "binary", pos_label = 0)
	gp_precision_neg += (1.0/k)*precision_score(y_test, gppred_test, average = "binary", pos_label = 0)
	gp_recall_neg += (1.0/k)*recall_score(y_test, gppred_test, average = "binary",pos_label = 0)
	gp_f1_neg += (1.0/k)*f1_score(y_test, gppred_test, average = "binary", pos_label = 0)
	gp_means.append(np.mean(gppred_test == y_test))
	print("**********************************")

	if (count == 1):
		gppred_prob = clf.predict_proba(X_test)
		fpr6, tpr6, thresholds6	 = roc_curve(y_test, gppred_prob[:,1])
		roc_auc6 = auc(fpr6, tpr6)
		plt.plot(fpr6, tpr6, label = 'GP (area = %0.2f)' % roc_auc6)

	print(count)
	count = count + 1
	# if (count == 2):
	# 	break
	
print("LR")
print("Accuracy: ", lr_accuracy)
print("Time: ", lr_time)
print("Precision: ", lr_precision)
print("Recall: ", lr_recall)
print("F1: ", lr_f1)
print("NB")
print("Accuracy: ", nb_accuracy)
print("Time: ", nb_time)
print("Precision: ", nb_precision)
print("Recall: ", nb_recall)
print("F1: ", nb_f1)
print("KNN")
print("Accuracy: ", knn_accuracy)
print("Time: ", knn_time)
print("Precision: ", knn_precision)
print("Recall: ", knn_recall)
print("F1: ", knn_f1)
print("RF")
print("Accuracy: ", rf_accuracy)
print("Time: ", rf_time)
print("Precision: ", rf_precision)
print("Recall: ", rf_recall)
print("F1: ", rf_f1)
print("DF")
print("Accuracy: ", dt_accuracy)
print("Time: ", dt_time)
print("Precision: ", dt_precision)
print("Recall: ", dt_recall)
print("F1: ", dt_f1)
print("GP")
print("Accuracy: ", gp_accuracy)
print("Time: ", gp_time)
print("Precision: ", gp_precision)
print("Recall: ", gp_recall)
print("F1: ", gp_f1)

import scipy.stats as st
print(st.t.interval(0.95, len(lr_means)-1, loc=np.mean(lr_means), scale=st.sem(lr_means)))
print(st.t.interval(0.95, len(knn_means)-1, loc=np.mean(knn_means), scale=st.sem(knn_means)))
print(st.t.interval(0.95, len(rf_means)-1, loc=np.mean(rf_means), scale=st.sem(rf_means)))
print(st.t.interval(0.95, len(gp_means)-1, loc=np.mean(gp_means), scale=st.sem(gp_means)))
# print(svm_accuracy)
# # 	# MULTIPLE IMPUTATION IF INSIDE THIS LOOP #
# 	# imputer = preprocessing.Imputer(strategy = "most_frequent")
# 	# df = imputer.fit(df)
# 	# training_data = imputer.transform(training_data)
# 	# test_data = imputer.transform(test_data)


# 	training_outcomes = [int(i) for i in training_outcomes]
# 	test_outcomes = [int(i) for i in test_outcomes]

# training_data[training_data<0] = 1
# ################# Feature Selection ####################
# selector = SelectPercentile(mutual_info_classif, percentile = 50).fit(training_data, training_outcomes)

# ##### Model Selection #####
# # # selector = SelectKBest(mutual_info_regression, k=10).fit(training_data, training_outcomes)
# lasso = GradientBoostingRegressor()
# selector = SelectFromModel(lasso)
# selector.fit(training_data, training_outcomes)

# idxs = selector.get_support(indices = True)
# training_data = selector.transform(training_data)
# test_data = test_data[:,idxs]






# # ################ CLASSIIFICATION #######################
# # ## BOOOOOOOT STRAAAAAAAPPPPPPP ##

# # outcomes_eviction = [int(i) for i in outcomes_eviction.challengeID]
# # # bcv = Bootstrap(n = len(outcomes_eviction))

# # # for index_train, index_test in bcv:

# # # 	X_train, X_test = df[index_train,:], df[index_test,:]

# # # 	#print(np.mean(X_test))
# # # 	y_train = [outcomes_eviction[i] for i in index_train]
# # # 	y_test = [outcomes_eviction[i] for i in index_test]



	# print("###### NB CLASSIFIER ######")
# clf  = LogisticRegression()
# clf.fit(training_data, training_outcomes)
# #nbpred_train = clf.predict(training_data)
# nbpred_test = clf.predict(test_data)
# print(training_data.shape)
# print(len(training_outcomes))
# print(test_data.shape)
# print(len(test_outcomes))

# print(len(nbpred_test))
# print(clf.score(test_data, test_outcomes))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves for Classifiers (eviction)')
# plt.legend(loc="lower right")
# plt.show()