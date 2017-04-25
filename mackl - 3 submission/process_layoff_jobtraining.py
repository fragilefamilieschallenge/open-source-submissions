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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import time

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.svm import SVC


import matplotlib.pyplot as plt
#from sklearn.cross_validation import Bootstrap

## NEWBACKGROUND.CSV GENERATED FROM R SCRIPT ##

from numpy import genfromtxt
df = genfromtxt('completedBackground.csv', delimiter=',')
#df = pd.read_csv("completedBackground.csv", low_memory=False, header = None)

print("Number of features")
print(df.shape)

# imputer = preprocessing.Imputer(strategy = "mean")
# df = imputer.fit_transform(df) 
# df[df < 0] = 1

# selector = VarianceThreshold()
# df = selector.fit_transform(df)
# print("Variance Threshold")
# print(df.shape)

outcomes = pd.read_csv("train.csv", low_memory=False)

# Get rows of training data that have reported jobTraining (exclude those with NA)
outcomes_jobTraining = outcomes[pd.notnull(outcomes['jobTraining'])]
target = [int(i) for i in outcomes_jobTraining.jobTraining]

### Begin Classification Process ###

from sklearn.model_selection import StratifiedKFold
k = 10
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

knn_accuracy = 0.0
knn_precision = 0
knn_recall = 0
knn_f1 = 0
knn_precision_neg = 0
knn_recall_neg = 0
knn_f1_neg = 0
knn_time = 0

rf_accuracy = 0.0
rf_precision = 0
rf_recall = 0
rf_f1 = 0
rf_precision_neg = 0
rf_recall_neg = 0
rf_f1_neg = 0
rf_time = 0

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

svm_accuracy = 0.0
svm_precision = 0
svm_recall = 0
svm_f1 = 0
svm_precision_neg = 0
svm_recall_neg = 0
svm_f1_neg = 0
svm_time = 0

print(df[0,:])
df[df < 0] = 1
for index_train, index_test in skf.split(df, target):
	#print(index_train)
	#print(index_test)
	X_train = df[index_train,:]
	X_test = df[index_test,:]
	y_train = [target[i] for i in index_train]
	y_test = [target[i] for i in index_test]


	print("******** Naive Bayes ******")
	clf = MultinomialNB()
	start_time = time.time()
	clf.fit(X_train, y_train)
	nbpred_test = clf.predict(X_test)
	nb_time += (time.time() - start_time)
	nb_accuracy += (1.0/k)*np.mean(nbpred_test == y_test)
	nb_precision += (1.0/k)*precision_score(y_test, nbpred_test, average = "binary", pos_label = 0)
	nb_recall += (1.0/k)*recall_score(y_test, nbpred_test, average = "binary", pos_label = 0)
	nb_f1 += (1.0/k)*f1_score(y_test, nbpred_test, average = "binary", pos_label = 0)
	nb_precision_neg += (1.0/k)*precision_score(y_test, nbpred_test, average = "binary", pos_label = 0)
	nb_recall_neg += (1.0/k)*recall_score(y_test, nbpred_test, average = "binary",pos_label = 0)
	nb_f1_neg += (1.0/k)*f1_score(y_test, nbpred_test, average = "binary", pos_label = 0)
	print("**********************************")

	# selector = SelectPercentile(chi2, percentile = 75).fit(X_train, y_train)
	# idxs = selector.get_support(indices = True)
	# X_train= selector.transform(X_train)
	# X_test = X_test[:,idxs]
	# print(X_train.shape)

	print("******** Logistic Regression ******")
	clf1 = LogisticRegression()
	clf1.fit(X_train, y_train)
	start_time = time.time()
	lrpred_test = clf1.predict(X_test)
	lr_time += (time.time() - start_time)
	lr_accuracy += (1.0/k)*np.mean(lrpred_test == y_test)
	lr_precision += (1.0/k)*precision_score(y_test, lrpred_test, average = "binary", pos_label = 0)
	lr_recall += (1.0/k)*recall_score(y_test, lrpred_test, average = "binary", pos_label = 0)
	lr_f1 += (1.0/k)*f1_score(y_test, lrpred_test, average = "binary", pos_label = 0)
	lr_precision_neg += (1.0/k)*precision_score(y_test, lrpred_test, average = "binary", pos_label = 0)
	lr_recall_neg += (1.0/k)*recall_score(y_test, lrpred_test, average = "binary",pos_label = 0)
	lr_f1_neg += (1.0/k)*f1_score(y_test, lrpred_test, average = "binary", pos_label = 0)
	print("**********************************")

	print("******** K-Nearest Neighbors ******")
	clf = KNeighborsClassifier(n_neighbors = 5, algorithm ='kd_tree')
	start_time = time.time()
	clf.fit(X_train, y_train)
	knnpred_test = clf.predict(X_test)
	knn_time += (time.time() - start_time)
	knn_accuracy += (1.0/k)*np.mean(knnpred_test == y_test)
	knn_precision += (1.0/k)*precision_score(y_test, knnpred_test, average = "binary", pos_label = 0)
	knn_recall += (1.0/k)*recall_score(y_test, knnpred_test, average = "binary", pos_label = 0)
	knn_f1 += (1.0/k)*f1_score(y_test, knnpred_test, average = "binary", pos_label = 0)
	knn_precision_neg += (1.0/k)*precision_score(y_test, knnpred_test, average = "binary", pos_label = 0)
	knn_recall_neg += (1.0/k)*recall_score(y_test, knnpred_test, average = "binary",pos_label = 0)
	knn_f1_neg += (1.0/k)*f1_score(y_test, knnpred_test, average = "binary", pos_label = 0)

	print("**********************************")

	print("******** Random Forest ******")
	clf = RandomForestClassifier(n_estimators = 100,criterion = "gini", max_depth = 10)
	start_time = time.time()
	clf.fit(X_train, y_train)
	rfpred_test = clf.predict(X_test)
	rf_time += (time.time() - start_time)
	rf_accuracy += (1.0/k)*np.mean(rfpred_test == y_test)
	rf_precision += (1.0/k)*precision_score(y_test, rfpred_test, average = "binary", pos_label = 0)
	rf_recall += (1.0/k)*recall_score(y_test, rfpred_test, average = "binary", pos_label = 0)
	rf_f1 += (1.0/k)*f1_score(y_test, rfpred_test, average = "binary", pos_label = 0)
	rf_precision_neg += (1.0/k)*precision_score(y_test, rfpred_test, average = "binary", pos_label = 0)
	rf_recall_neg += (1.0/k)*recall_score(y_test, rfpred_test, average = "binary",pos_label = 0)
	rf_f1_neg += (1.0/k)*f1_score(y_test, rfpred_test, average = "binary", pos_label = 0)
	print("**********************************")

	print("******** Decision Tree ******")
	clf = DecisionTreeClassifier(criterion = "gini")
	start_time = time.time()
	clf.fit(X_train, y_train)
	dtpred_test = clf.predict(X_test)
	dt_time += (time.time() - start_time)
	dt_accuracy += (1.0/k)*np.mean(dtpred_test == y_test)
	dt_precision += (1.0/k)*precision_score(y_test, dtpred_test, average = "binary", pos_label = 0)
	dt_recall += (1.0/k)*recall_score(y_test, dtpred_test, average = "binary", pos_label = 0)
	dt_f1 += (1.0/k)*f1_score(y_test, dtpred_test, average = "binary", pos_label = 0)
	dt_precision_neg += (1.0/k)*precision_score(y_test, dtpred_test, average = "binary", pos_label = 0)
	dt_recall_neg += (1.0/k)*recall_score(y_test, dtpred_test, average = "binary",pos_label = 0)
	dt_f1_neg += (1.0/k)*f1_score(y_test, dtpred_test, average = "binary", pos_label = 0)
	print("**********************************")

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
	# print("**********************************")
	
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
	print("**********************************")

	print(count)
	count = count + 1
	
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
print("SVM")
print("Accuracy: ", svm_accuracy)
print("Time: ", svm_time)
print("Precision: ", svm_precision)
print("Recall: ", svm_recall)
print("F1: ", svm_f1)
print("GP")
print("Accuracy: ", gp_accuracy)
print("Time: ", gp_time)
print("Precision: ", gp_precision)
print("Recall: ", gp_recall)
print("F1: ", gp_f1)
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


# # # from sklearn import preprocessing
# # # scaler = preprocessing.StandardScaler()
# # # normalized_training_data = scaler.fit_transform(training_data)
# # # normalized_test_data = scaler.fit_transform(test_data)



# # ################ CLASSIIFICATION #######################
# # ## BOOOOOOOT STRAAAAAAAPPPPPPP ##

# # outcomes_jobTraining = [int(i) for i in outcomes_jobTraining.challengeID]
# # # bcv = Bootstrap(n = len(outcomes_jobTraining))

# # # for index_train, index_test in bcv:

# # # 	X_train, X_test = df[index_train,:], df[index_test,:]

# # # 	#print(np.mean(X_test))
# # # 	y_train = [outcomes_jobTraining[i] for i in index_train]
# # # 	y_test = [outcomes_jobTraining[i] for i in index_test]



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
