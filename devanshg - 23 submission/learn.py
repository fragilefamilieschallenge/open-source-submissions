from __future__ import division
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score, f1_score, precision_score, confusion_matrix, precision_recall_curve, mean_squared_error
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, LogisticRegression, LassoLars, RANSACRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelBinarizer
import time
import math
import csv
import warnings

def feature_selection_imp(train_matrix, test_matrix, train_targets, dictionary_file):
	clf = ExtraTreesClassifier()
	clf = clf.fit(train_matrix, train_targets)
	imp = clf.feature_importances_
	# for i in xrange(len(imp)):
	#  	print imp[i]

	num = xrange(len(imp))
	vocab = []
	with open(dictionary_file, "r") as f:
		for line in f:
			vocab.append(line)
	numberedImp = zip(imp, num, vocab)

	# Change threshold to vary number of features selected
	threshold = 0.0000000000000000

	numsToDel = []
	wordsToDel = []
	for i in xrange(len(numberedImp)):
		val, no, word = numberedImp[i]
		if val < threshold:
			numsToDel.append(no)
			wordsToDel.append(word)

	print wordsToDel
	print len(wordsToDel)

	train_matrix = SelectFromModel(clf, threshold= threshold, prefit=True).transform(train_matrix)
	test_matrix = SelectFromModel(clf, threshold= threshold, prefit=True).transform(test_matrix)

	num_features = len(train_matrix[0])
	# print len(train_matrix[0])
	# print len(test_matrix[0])

	return train_matrix, test_matrix, num_features

def feature_selection_chi2(train_matrix, test_matrix, train_targets, k_best, headers):
    k = SelectKBest(chi2, k=k_best)
    train_matrix = k.fit_transform(train_matrix, train_targets)
    test_matrix = k.transform(test_matrix)
    num_features = len(test_matrix[0])
    indices = k.get_support(indices=True)
    # print "Features Selected Chi2:"
    # for i in indices:
    #     print headers[i] + " "
    return train_matrix, test_matrix, num_features

def feature_selection_f_classif(train_matrix, test_matrix, train_targets, k_best, headers):
    k = SelectKBest(k=k_best)
    train_matrix = k.fit_transform(train_matrix, train_targets)
    test_matrix = k.transform(test_matrix)
    num_features = len(test_matrix[0])
    indices = k.get_support(indices=True)
    # print "Features Selected F Classif:"
    # for i in indices:
    #     print headers[i] + " "
    return train_matrix, test_matrix, num_features

def prediction_Error_Bootstrap(model, train_matrix, train_targets):
    mse_list = []
    for i in xrange(1000):
        X_train, X_test, y_train, y_test = train_test_split(train_matrix, train_targets, test_size=0.33, random_state=i)
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        # print "Predicted:"
        # print test_predictions[0:10]
        # print "Actual:"
        # print y_test[0:10]
        mse = np.mean(np.square(y_test - test_predictions))
        mse_list.append(mse)

    # print mse_list
    mse_list.sort()
    return mse_list[24], mse_list[974]

def run_multi_classifications(X, y, X_test, labelname, k, features, headers):
    ret_predictions = {}

    if features != -1:
        y_new = np.multiply(y, 100).astype(int)
        X, X_test, n = feature_selection_chi2(X, X_test, y_new, features, headers)
    print('{} : {}'.format("Feature Selected X", X.shape))
    print('{} : {}'.format("Feature Selected X_test", X_test.shape))
    print_line()

    if labelname == 'Material Hardship':
        y = np.multiply(y, 11)
        y = np.add(y, 1)
        y = y.astype(int)
    else:
        y = np.multiply(y, 100)
        y = y.astype(int)

    lb = LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)

    k_fold = KFold(n_splits=k, shuffle=True, random_state=0)

    ## L2 OVR LOGISTIC REGRESSION ######
    #lr2o = LogisticRegression(multi_class='ovr')
    #start_time = time.time()
    #lr2o.fit(X, y)
    #runtime = str(time.time() - start_time)
    #y_train = lr2o.predict(X)
    #y_test = lr2o.predict(X_test)
    #print_classification_stats("L2 OVR Logistic Regression " + labelname, y, y_train, y_test, runtime)
    #cv = cross_val_score(lr2o, X, y, cv=k_fold)
    #print "CV Score: " + str(cv)
    #print "CV Average: " + str(sum(cv)/float(len(cv)))
    #print_line()
    #y_train = lb.inverse_transform(y_train)
    #y_test = lb.inverse_transform(y_test)
    #ret_predictions['lr2o'] = np.concatenate((y_train, y_test))

    # L2 Multinomial LOGISTIC REGRESSION ######
    #lr2m = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    #start_time = time.time()
    #lr2m.fit(X, y)
    #runtime = str(time.time() - start_time)
    #y_train = lr2m.predict(X)
    #y_test = lr2m.predict(X_test)
    #print_classification_stats("L2 Multinomial Logistic Regression " + labelname, y, y_train, y_test, runtime)
    #cv = cross_val_score(lr2m, X, y, cv=k_fold)
    #print "CV Score: " + str(cv)
    #print "CV Average: " + str(sum(cv)/float(len(cv)))
    #print_line()
    #y_train = lb.inverse_transform(y_train)
    #y_test = lb.inverse_transform(y_test)
    #ret_predictions['lr2m'] = np.concatenate((y_train, y_test))

    # K NEAREST NEIGHBORS ######
    neigh = KNeighborsClassifier(20)
    start_time = time.time()
    neigh.fit(X, y)
    runtime = str(time.time() - start_time)
    y_train = neigh.predict(X)
    y_test = neigh.predict(X_test)
    print_classification_stats("KNN Multi " + labelname, y, y_train, y_test, runtime)
    cv = cross_val_score(neigh, X, y, cv=k_fold, scoring='mean_squared_error')
    print "CV Score: " + str(cv)
    print "CV Average: " + str(sum(cv)/float(len(cv)))
    print_line()
    y_train = lb.inverse_transform(y_train)
    y_test = lb.inverse_transform(y_test)
    ret_predictions['knn'] = np.concatenate((y_train, y_test))

    ## L1 OVR LOGISTIC REGRESSION ######
    #lr1o = LogisticRegression(multi_class='ovr', penalty='l1')
    #start_time = time.time()
    #lr1o.fit(X, y)
    #runtime = str(time.time() - start_time)
    #y_train = lr1o.predict(X)
    #y_test = lr1o.predict(X_test)
    #print_classification_stats("L1 OVR Logistic Regression " + labelname, y, y_train, y_test, runtime)
    #cv = cross_val_score(lr1o, X, y, cv=k_fold)
    #print "CV Score: " + str(cv)
    #print "CV Average: " + str(sum(cv)/float(len(cv)))
    #print_line()
    #y_train = lb.inverse_transform(y_train)
    #y_test = lb.inverse_transform(y_test)
    #ret_predictions['lr1o'] = np.concatenate((y_train, y_test))

    # liblinear solver doesnt support a multinomial backend. But the other solvers don't work with l1 penalty. So...

    ## L1 Multinomial LOGISTIC REGRESSION ######
    #lr1m = LogisticRegression(multi_class='multinomial', penalty='l1')
    #start_time = time.time()
    #lr1m.fit(X, y)
    #runtime = str(time.time() - start_time)
    #y_train = lr1m.predict(X)
    #y_test = lr1m.predict(X_test)
    #print_classification_stats("L1 Multinomial Logistic Regression " + labelname, y, y_train, y_test, runtime)
    #cv = cross_val_score(lr1m, X, y, cv=k_fold)
    #print "CV Score: " + str(cv)
    #print "CV Average: " + str(sum(cv)/float(len(cv)))
    #print_line()
    #y_train = lb.inverse_transform(y_train)
    #y_test = lb.inverse_transform(y_test)
    #ret_predictions['lr1m'] = np.concatenate((y_train, y_test))

    return ret_predictions

def run_classifications(X, y, X_test, labelname, k, features, headers):
    ret_predictions = {}

    if features != -1:
        X1, X_test1, n1 = feature_selection_chi2(X, X_test, y, features, headers)
        X2, X_test2, n2 = feature_selection_f_classif(X, X_test, y, features, headers)
        X = np.concatenate((X1,X2), axis=1)
        X_test = np.concatenate((X_test1,X_test2), axis=1)

    # X, X_test, n = feature_selection_f_classif(X, X_test, y, features, headers)

    print('{} : {}'.format("Feature Selected X", X.shape))
    print('{} : {}'.format("Feature Selected X_test", X_test.shape))
    print_line()

    # return

    k_fold = KFold(n_splits=k, shuffle=True, random_state=0)

    # # L2 LOGISTIC REGRESSION ######
    # lr2 = LogisticRegression()
    # start_time = time.time()
    # lr2.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = lr2.predict(X)
    # y_test = lr2.predict(X_test)
    # print_classification_stats("L2 Logistic Regression " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(lr2, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # print_line()
    # ret_predictions['lr2'] = np.concatenate((y_train, y_test))

    # # L1 LOGISTIC REGRESSION ######
    # lr1 = LogisticRegression(penalty='l1')
    # start_time = time.time()
    # lr1.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = lr1.predict(X)
    # y_test = lr1.predict(X_test)
    # print_classification_stats("L2 Logistic Regression " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(lr1, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # print_line()
    # ret_predictions['lr1'] = np.concatenate((y_train, y_test))

    # RANDOM FOREST ######
    rf = RandomForestClassifier()
    start_time = time.time()
    rf.fit(X, y)
    runtime = str(time.time() - start_time)
    y_train = rf.predict(X)
    y_test = rf.predict(X_test)
    print_classification_stats("Random Forest " + labelname, y, y_train, y_test, runtime)
    cv = cross_val_score(rf, X, y, cv=k_fold, scoring='mean_squared_error')
    print "CV Score: " + str(cv)
    print "CV Average: " + str(sum(cv)/float(len(cv)))
    print_line()
    ret_predictions['rf'] = np.concatenate((y_train, y_test))

    # lo,hi = prediction_Error_Bootstrap(rf, X, y)
    # print ".95 Confidence Interval: " + str(lo) + " - " + str(hi)

    # # K NEAREST NEIGHBORS ######
    # neigh = KNeighborsClassifier(4)
    # start_time = time.time()
    # neigh.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = neigh.predict(X)
    # y_test = neigh.predict(X_test)
    # print_classification_stats("KNN " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(neigh, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # print_line()
    # ret_predictions['knn'] = np.concatenate((y_train, y_test))

    # lo,hi = prediction_Error_Bootstrap(neigh, X, y)
    # print ".95 Confidence Interval: " + str(lo) + " - " + str(hi)

    # # Linear SVM ######
    # #svc = SVC(kernel='linear', C=0.025)
    # #start_time = time.time()
    # #svc.fit(X, y)
    # #runtime = str(time.time() - start_time)
    # #y_train = svc.predict(X)
    # #y_test = svc.predict(X_test)
    # #print_classification_stats("Linear SVM " + labelname, y, y_train, y_test, runtime)
    # #cv = cross_val_score(svc, X, y, cv=k_fold, scoring='mean_squared_error')
    # #print "CV Score: " + str(cv)
    # #print "CV Average: " + str(sum(cv)/float(len(cv)))
    # #print_line()
    # #ret_predictions['svc'] = np.concatenate((y_train, y_test))

    # ## RBF SVM ######
    # #rsvc = SVC(gamma=2, C=1)
    # #start_time = time.time()
    # #rsvc.fit(X, y)
    # #runtime = str(time.time() - start_time)
    # #y_train = rsvc.predict(X)
    # #y_test = rsvc.predict(X_test)
    # #print_classification_stats("RBF SVM " + labelname, y, y_train, y_test, runtime)
    # #cv = cross_val_score(rsvc, X, y, cv=k_fold, scoring='mean_squared_error')
    # #print "CV Score: " + str(cv)
    # #print "CV Average: " + str(sum(cv)/float(len(cv)))
    # #print_line()
    # #ret_predictions['rbf'] = np.concatenate((y_train, y_test))

    # Gaussian Process ######
    gp = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
    start_time = time.time()
    gp.fit(X, y)
    runtime = str(time.time() - start_time)
    y_train = gp.predict(X)
    y_test = gp.predict(X_test)
    print_classification_stats("Gaussian Process " + labelname, y, y_train, y_test, runtime)
    cv = cross_val_score(gp, X, y, cv=k_fold, scoring='mean_squared_error')
    print "CV Score: " + str(cv)
    print "CV Average: " + str(sum(cv)/float(len(cv)))
    print_line()
    ret_predictions['gp'] = np.concatenate((y_train, y_test))

    # lo,hi = prediction_Error_Bootstrap(gp, X, y)
    # print ".95 Confidence Interval: " + str(lo) + " - " + str(hi)

    # # Decision Tree ######
    # dt = DecisionTreeClassifier()
    # start_time = time.time()
    # dt.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = dt.predict(X)
    # y_test = dt.predict(X_test)
    # print_classification_stats("Decision Tree " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(dt, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # print_line()
    # ret_predictions['dt'] = np.concatenate((y_train, y_test))

    # # Neural Net ######
    # mlp = MLPClassifier()
    # start_time = time.time()
    # mlp.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = mlp.predict(X)
    # y_test = mlp.predict(X_test)
    # print_classification_stats("Neural Net " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(mlp, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # print_line()
    # ret_predictions['mlp'] = np.concatenate((y_train, y_test))

    # # AdaBoost Classifier ######
    # ab = AdaBoostClassifier()
    # start_time = time.time()
    # ab.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = ab.predict(X)
    # y_test = ab.predict(X_test)
    # print_classification_stats("AdaBoost " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(ab, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # print_line()
    # ret_predictions['ab'] = np.concatenate((y_train, y_test))

    # lo,hi = prediction_Error_Bootstrap(ab, X, y)
    # print ".95 Confidence Interval: " + str(lo) + " - " + str(hi)

    # # Naive Bayes ######
    # gnb = GaussianNB()
    # start_time = time.time()
    # gnb.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = gnb.predict(X)
    # y_test = gnb.predict(X_test)
    # print_classification_stats("Naive Bayes " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(gnb, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # print_line()
    # ret_predictions['gnb'] = np.concatenate((y_train, y_test))

    # # QDA ######
    # qda = QuadraticDiscriminantAnalysis()
    # start_time = time.time()
    # qda.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = qda.predict(X)
    # y_test = qda.predict(X_test)
    # print_classification_stats("QDA " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(qda, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # print_line()
    # ret_predictions['qda'] = np.concatenate((y_train, y_test))

    # lo,hi = prediction_Error_Bootstrap(qda, X, y)
    # print ".95 Confidence Interval: " + str(lo) + " - " + str(hi)

    return ret_predictions

def run_regressions(X, y, X_test, labelname, k, features, headers):

    ret_predictions = {}

    if features != -1:
        y_new = np.multiply(y, 100).astype(int)
        X1, X_test1, n1 = feature_selection_chi2(X, X_test, y_new, features, headers)
        X2, X_test2, n2 = feature_selection_f_classif(X, X_test, y_new, features, headers)
        X = np.concatenate((X1,X2), axis=1)
        X_test = np.concatenate((X_test1,X_test2), axis=1)
    
    # y_new = np.multiply(y, 100).astype(int)
    # X, X_test, n = feature_selection_f_classif(X, X_test, y_new, features, headers)
    print('{} : {}'.format("Feature Selected X", X.shape))
    print('{} : {}'.format("Feature Selected X_test", X_test.shape))
    print_line()

    # return

    # gnb = GaussianNB()
    # start_time = time.time()
    # gnb.fit(train_matrix, train_targets)
    # runtime = str(time.time() - start_time)
    # y_train = gnb.predict(train_matrix)
    # y_test = gnb.predict(test_matrix)

    # print_stats("Gaussian Naive Bayes", train_targets, test_targets, y_train, y_test, runtime, num_features)

    # y_prob = gnb.predict_proba(test_matrix)
    # fpr, tpr, thresholds = roc_curve(test_targets, y_prob[:,1])
    # per, rec, thresh = precision_recall_curve(test_targets, y_prob[:,1])
    # # pr_auc = auc(per, rec)
    # roc_auc = auc(fpr, tpr)

    # plt.plot(fpr, tpr, lw=2, color='#83b2d0',label='Gaussian Naive Bayes ROC (area = %0.2f)' % (roc_auc))

    # pers.append(per)
    # recalls.append(rec)
    # threshs.append(thresh)
    # # pr_aucs.append(pr_auc)

    k_fold = KFold(n_splits=k, shuffle=True, random_state=0)

    # Linear Regression ######
    # lr = LinearRegression()
    # start_time = time.time()
    # lr.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = lr.predict(X)
    # #print y_train[:100]
    # y_test = lr.predict(X_test)
    # print_regression_stats("Linear Regression " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(lr, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # print_line()
    # ret_predictions['lr'] = np.concatenate((y_train, y_test))

    # KNN Regression ######
    knn = KNeighborsRegressor(23)
    start_time = time.time()
    knn.fit(X, y)
    runtime = str(time.time() - start_time)
    y_train = knn.predict(X)
    y_test = knn.predict(X_test)
    print_regression_stats("KNN Regression " + labelname, y, y_train, y_test, runtime)
    cv = cross_val_score(knn, X, y, cv=k_fold, scoring='mean_squared_error')
    print "CV Score: " + str(cv)
    print "CV Average: " + str(sum(cv)/float(len(cv)))
    ret_predictions['knn'] = np.concatenate((y_train, y_test))

    # lo,hi = prediction_Error_Bootstrap(knn, X, y)
    # print ".95 Confidence Interval: " + str(lo) + " - " + str(hi)
    print_line()

    # Epsilon-Support Vector Regression ######
    svr = SVR()
    start_time = time.time()
    svr.fit(X, y)
    runtime = str(time.time() - start_time)
    y_train = svr.predict(X)
    y_test = svr.predict(X_test)
    print_regression_stats("SVR " + labelname, y, y_train, y_test, runtime)
    cv = cross_val_score(svr, X, y, cv=k_fold, scoring='mean_squared_error')
    print "CV Score: " + str(cv)
    print "CV Average: " + str(sum(cv)/float(len(cv)))
    ret_predictions['svr'] = np.concatenate((y_train, y_test))

    # lo,hi = prediction_Error_Bootstrap(svr, X, y)
    # print ".95 Confidence Interval: " + str(lo) + " - " + str(hi)
    print_line()

    # # Lasso LARS Regression ######
    # ll = LassoLars()
    # start_time = time.time()
    # ll.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = ll.predict(X)
    # y_test = ll.predict(X_test)
    # print_regression_stats("Lasso Lars " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(ll, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # ret_predictions['ll'] = np.concatenate((y_train, y_test))

    # lo,hi = prediction_Error_Bootstrap(ll, X, y)
    # print ".95 Confidence Interval: " + str(lo) + " - " + str(hi)
    # print_line()

    # # Kernel Ridge Regression ######
    # kr = KernelRidge()
    # start_time = time.time()
    # kr.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = kr.predict(X)
    # y_test = kr.predict(X_test)
    # print_regression_stats("Kernel Ridge " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(kr, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # ret_predictions['kr'] = np.concatenate((y_train, y_test))

    # lo,hi = prediction_Error_Bootstrap(kr, X, y)
    # print ".95 Confidence Interval: " + str(lo) + " - " + str(hi)
    # print_line()

    # Ridge Regression ######
    # r = Ridge()
    # start_time = time.time()
    # r.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = r.predict(X)
    # y_test = r.predict(X_test)
    # print_regression_stats("Ridge " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(r, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # print_line()
    # ret_predictions['r'] = np.concatenate((y_train, y_test))

    # Lasso Regression ######
    l = Lasso()
    start_time = time.time()
    l.fit(X, y)
    runtime = str(time.time() - start_time)
    y_train = l.predict(X)
    y_test = l.predict(X_test)
    print_regression_stats("Lasso " + labelname, y, y_train, y_test, runtime)
    cv = cross_val_score(l, X, y, cv=k_fold, scoring='mean_squared_error')
    print "CV Score: " + str(cv)
    print "CV Average: " + str(sum(cv)/float(len(cv)))
    ret_predictions['l'] = np.concatenate((y_train, y_test))

    # lo,hi = prediction_Error_Bootstrap(l, X, y)
    # print ".95 Confidence Interval: " + str(lo) + " - " + str(hi)
    print_line()

    # Elastic Net ######
    # el = ElasticNet()
    # start_time = time.time()
    # el.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = el.predict(X)
    # y_test = el.predict(X_test)
    # print_regression_stats("Elastic Net " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(el, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # print_line()
    # ret_predictions['el'] = np.concatenate((y_train, y_test))

    # Bayesian Ridge ######
    # br = BayesianRidge()
    # start_time = time.time()
    # br.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = br.predict(X)
    # y_test = br.predict(X_test)
    # print_regression_stats("Bayesian Ridge " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(br, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # print_line()
    # ret_predictions['br'] = np.concatenate((y_train, y_test))

    # # Gaussian Process ######
    # gp = GaussianProcessRegressor()
    # start_time = time.time()
    # gp.fit(X, y)
    # runtime = str(time.time() - start_time)
    # y_train = gp.predict(X)
    # y_test = gp.predict(X_test)
    # print_regression_stats("Gaussian Process " + labelname, y, y_train, y_test, runtime)
    # cv = cross_val_score(gp, X, y, cv=k_fold, scoring='mean_squared_error')
    # print "CV Score: " + str(cv)
    # print "CV Average: " + str(sum(cv)/float(len(cv)))
    # ret_predictions['gp'] = np.concatenate((y_train, y_test))

    # lo,hi = prediction_Error_Bootstrap(gp, X, y)
    # print ".95 Confidence Interval: " + str(lo) + " - " + str(hi)
    # print_line()

    return ret_predictions

def print_line():
    print "------------------------------------------------------------------------"


def print_classification_stats(name, y, y_train, y_test, runtime, num_features=None):

    if num_features is not None:
        print "Number of features selected: " + str(num_features)

    # Accuracy score
    print name + " Train Accuracy: " + str(accuracy_score(y, y_train))
    # print name + " Test Accuracy: " + str(accuracy_score(test_targets, y_test))

    # Precision
    # print name + " Train Precision Score: " + str(precision_score(y, y_train))
    # print name + " Test Precision Score: " + str(precision_score(test_targets, y_test))

    # Recall
    # print name + " Train Recall Score: " + str(recall_score(y, y_train))
    # print name + " Test Recall Score: " + str(recall_score(test_targets, y_test))

    # F1
    # print name + " Train F1 Score: " + str(f1_score(y, y_train))
    # print name + " Test F1 Score: " + str(f1_score(test_targets, y_test))

    # Runtime
    print name + " Fitting Runtime: " + runtime

def print_regression_stats(name, y, y_train, y_test, runtime, num_features=None):

    if num_features is not None:
        print "Number of features selected: " + str(num_features)

    # Accuracy score
    print name + " Mean Squared Error: " + str(mean_squared_error(y, y_train))

    # Runtime
    print name + " Fitting Runtime: " + runtime

def write_predictions(rows):
    with open('prediction_reg.csv', 'wb') as f:
        w = csv.writer(f)
        w.writerow(("challengeID", "gpa", "grit", "materialHardship", "eviction", "layoff", "jobTraining"))
        for row in rows:
            w.writerow(row)

def revert_grit_gpa(p):
    y = p.astype(float)
    y = np.divide(y, float(100))
    return y

def revert_mhard(p):
    y = p.astype(float)
    y = np.subtract(y, 1)
    y = np.divide(y, float(11))
    return y

def roundoff_grit(p):
    p_new = np.copy(p)
    i = 0
    for s in p:
        if s < 1.25:
            p_new[i] = 1.25
        elif s > 4:
            p_new[i] = 4
        else:
            s = int((s + 0.125) / 0.25)
            p_new[i] = (s * 0.25)
        i = i + 1
    print p[:100]
    print p_new[:100]
    return p_new

def roundoff_gpa(p):
    p_new = np.copy(p)
    i = 0
    for s in p:
        if s < 1:
            p_new[i] = 1
        elif s > 4:
            p_new[i] = 4
        else:
            s = int((s + 0.125) / 0.25)
            p_new[i] = (s * 0.25)
        i = i + 1
    print p[:100]
    print p_new[:100]
    return p_new

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mc', default=0, type=int)
    parser.add_argument('--c', default=0, type=int)
    parser.add_argument('--r', default=0, type=int)
    parser.add_argument('--w', default=0, type=int)
    parser.add_argument('--k', default=2, type=int)
    parser.add_argument('--f', default=-1, type=int) # default: no feature selection
    parser.add_argument('--ro', default=0, type=int) # round off regression scores for gpa and grit
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    multinomial_classify = args.mc
    classify = args.c
    regress = args.r
    write = args.w
    k = args.k
    features = args.f
    roundoff = args.ro

    if multinomial_classify and regress and write:
        print "WARNING: Will only write results from Multinomial Classification!"

    try:
        X = np.load('X.npy')
        X_test = np.load('X_test.npy')
        y = np.load('y.npy')
        cID = np.load('cID.npy')
        headers = np.load('headers.npy')
        print "Loaded Train/Test Data from memory..."
    except IOError:
        X = np.genfromtxt('bg_train.csv', delimiter=',', dtype=float)
        X_test = np.genfromtxt('bg_test.csv', delimiter=',', dtype=float)
        y = np.genfromtxt('train_labels_filled.csv', delimiter=',')

        print "Length of y: " + str(len(y))
        na_rows = np.load('na_rows.npy')
        y = np.delete(y, na_rows, axis=0)
        print "Length of y after deletion: " + str(len(y))

        bg = open('background_filled.csv', 'r')
        headers = bg.readline().split(',')
        headers = np.asarray(headers)

        cID = np.concatenate((X[:,-1],X_test[:,-1]))
        print cID.shape
        print cID[:100]

        print('{} : {}'.format("Shape of X", X.shape))
        print('{} : {}'.format("Shape of y", y.shape))
        # print('{} : {}'.format("Shape of y_label", y_grit.shape))
        print('{} : {}'.format("Shape of X_test", X_test.shape))
        print('{} : {}'.format("Shape of headers", headers.shape))

        print('Removing bad columns')
        cols = []
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if math.isnan(X[i,j]):
                    if j not in cols:
                        cols.append(j)
                        #print X[i,j]
                        #print ('{},{}'.format(i, j))

        for i in range(X_test.shape[0]):
            for j in range(X_test.shape[1]):
                if math.isnan(X_test[i,j]):
                    if j not in cols:
                        cols.append(j)
                        #print X[i,j]
                        #print ('{},{}'.format(i, j))
        X = np.delete(X, cols, axis=1)
        X_test = np.delete(X_test, cols, axis=1)
        headers = np.delete(headers, cols)
        print('Removed ' + str(len(cols)) + ' cols')
        print('{} : {}'.format("Final shape of X", X.shape))
        print('{} : {}'.format("Final shape of X_test", X_test.shape))
        print('{} : {}'.format("Final shape of headers", headers.shape))

        np.save('X.npy', X)
        np.save('X_test.npy', X_test)
        np.save('y.npy', y)
        np.save('cID.npy', cID)
        np.save('headers.npy', headers)
        print "Saved Train/Test Data to memory..."

    #X = X[0:X.shape[0], 100:300]
    #X_test = X_test[0:X_test.shape[0], 100:300]

    y_grit = y[:,1]
    y_gpa = y[:,2]
    y_mhardship = y[:,3]
    y_eviction = y[:,4]
    y_jobloss = y[:,5]
    y_jobtraining = y[:,6]

    p_evict = []
    p_jobloss = []
    p_jobtrain = []
    p_grit = []
    p_gpa = []
    p_mhard = []

    if classify:
        print "---------------------------------- CLASSIFICATIONS --------------------------------------"
        print("-----------------Eviction-----------------------------------------------")
        predicts = run_classifications(X, y_eviction, X_test, "Eviction", k, 1000, headers)
        p_evict = predicts['rf']
        print("-----------------Job Loss-----------------------------------------------")
        predicts = run_classifications(X, y_jobloss, X_test, "Job Loss", k, 1000, headers)
        p_jobloss = predicts['gp']
        print("---------------Job Training---------------------------------------------")
        predicts = run_classifications(X, y_jobtraining, X_test, "Job Training", k, 1000, headers)
        p_jobtrain = predicts['gp']
    if regress:
        print "----------------------------------   REGRESSIONS   --------------------------------------"
        print("----------------------Grit----------------------------------------------")
        predicts = run_regressions(X, y_grit, X_test, "Grit", k, 10, headers)
        p_grit = predicts['l']
        if roundoff:
            p_grit = roundoff_grit(p_grit)
        print("----------------------GPA-----------------------------------------------")
        predicts = run_regressions(X, y_gpa, X_test, "GPA", k, 23, headers)
        p_gpa = predicts['svr']
        if roundoff:
            p_gpa = roundoff_gpa(p_gpa)
        print("---------------Material Hardship----------------------------------------")
        predicts = run_regressions(X, y_mhardship, X_test, "Material Hardship", k, 10, headers)
        p_mhard = predicts['l']
    if multinomial_classify:
        print "------------------------------- MULTI CLASSIFICATIONS -----------------------------------"
        print("----------------------Grit----------------------------------------------")
        predicts = run_multi_classifications(X, y_grit, X_test, "Grit", k, features, headers)
        p_grit = predicts['knn'] #lr2m
        p_grit = revert_grit_gpa(p_grit)
        print p_grit[:100]
        print("----------------------GPA-----------------------------------------------")
        predicts = run_multi_classifications(X, y_gpa, X_test, "GPA", k, features, headers)
        p_gpa = predicts['knn']
        p_gpa = revert_grit_gpa(p_gpa)
        print p_gpa[:100]
        print("---------------Material Hardship----------------------------------------")
        predicts = run_multi_classifications(X, y_mhardship, X_test, "Material Hardship", k, features, headers)
        p_mhard = predicts['knn']
        p_mhard = revert_mhard(p_mhard)
        print p_mhard[:100]

    if len(p_evict) == 0:
        p_evict = np.ones(len(p_grit))
        p_jobloss = np.ones(len(p_grit))
        p_jobtrain = np.ones(len(p_grit))
    if len(p_grit) == 0:
        p_grit = np.ones(len(p_evict))
        p_gpa = np.ones(len(p_evict))
        p_mhard = np.ones(len(p_evict))

    if write:
        zipped = zip(cID, p_gpa, p_grit, p_mhard, p_evict, p_jobloss, p_jobtrain)
        zipped.sort()
        print_line()
        print "Writing predictions"
        write_predictions(zipped)
main()
