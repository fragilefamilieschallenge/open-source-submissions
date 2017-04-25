import csv
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from time import clock
import pdb
import pickle

prefix_list = ['Processing_Missing_Data_By_Just_Mean.txt', 'Processing_Missing_Data_By_Just_Mean_Threshold02.txt', 'Processing_Missing_Data_By_Just_Mean_Threshold06.txt', 'Whole_Dataset_Processing_Missing_Point','Whole_Dataset_Processing_Missing_Point_02Threshold', 'Whole_Dataset_Processing_Missing_Point_06Threshold']
for pfi in range(6):
    print pfi
    prefix = prefix_list[pfi]
    testclass = 'jobTraining'
    with open('train.csv','r') as train_file:
        train_reader = csv.DictReader(train_file)
        train_data = list(train_reader)

    trow = len(train_data)
    tcol = len(train_data[0])
    trainMatrix = np.zeros([trow,1])
    trainTag = []
    trainMatrix = trainMatrix - 1
    print "Number of train records: {}".format(len(train_data))
    #useful_rows = []
    for i in range(trow):
        gpa_item = train_data[i][testclass]
        try:
            if gpa_item == 'TRUE':
                trainMatrix[i] = 1
                trainTag.append(int(train_data[i]['challengeID']))
            elif gpa_item == 'FALSE':
                trainMatrix[i] = 0
                trainTag.append(int(train_data[i]['challengeID']))                
    #        useful_rows.append(float(trainTag[i]))
        except ValueError:
            pass

    trainMatrix = trainMatrix[trainMatrix>=0]
    for i in range(len(trainTag)):
        trainTag[i] = trainTag[i] - 1

    dataMatrix = np.load(prefix+'.npy')
    traindataMatrix = dataMatrix[trainTag]
    #[row, col] = traindataMatrix.shape
    '''for j in range(col):
        flag = False
        for i in range(row):
            if dataMatrix[i][j] < 0:
                flag =True
                break
        
        if flag:
            sumAll = 0
            counter = 0
            for i in range(row):
                if dataMatrix[i][j] >=0:
                    sumAll = sumAll + dataMatrix[i][j]
                    counter = counter + 1
            if counter == 0:
                avg = 0
            else:
                avg = float(sumAll)/float(counter)
            for i in range(row):
                if(dataMatrix[i][j] <0 ):
                    dataMatrix[i][j] = avg
    '''

    '''
    counter = 0
    row, col = dataMatrix.shape
    for j in range(col):
        flag = False
        for i in range(row):
            if dataMatrix[i][j] >= 0:
                flag = True
                break;
        if not flag:
            counter = counter + 1

    print counter
    print dataMatrix
    '''
    prefix = prefix_list[pfi] + 'JobTraining'
    stdsc = StandardScaler()
    '''

    X_train = traindataMatrix
    y_train = trainMatrix
    X_test = dataMatrix

    #X_train, X_test, y_train, y_test = train_test_split(dataMatrix, trainMatrix, test_size = 0.2)
    X_train_std = stdsc.fit_transform(X_train)
    #X_test_std = stdsc.transform(X_test)
    X_test_std = stdsc.transform(X_test)
    pca = PCA(n_components = 50)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    #print pca.components_
    #X_test_pca = pca.transform(X_test_std)
    #pdb.set_trace()

    '''
    '''
    print X_test_std

    print y_train
    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape
    print X_train_pca.shape
    print X_test_pca.shape
    '''
    '''
    clf = SVC()
    clf.fit(X_train_pca,y_train)      
    y = clf.predict(X_test_pca)    
    #print y
    np.save(prefix + '_SVM_pca.npy', y)

    clf = SVC()
    clf.fit(X_train_std,y_train)      
    y = clf.predict(X_test_std)
    #print y
    np.save(prefix + '_SVM_std.npy', y)
    #for i in range(result_row):
    #    sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    #print 'variance of Linear Regression Model is' + str(sumsum/result_row)

    clf = KNeighborsClassifier(n_neighbors = 10)
    clf.fit(X_train_pca,y_train)
    y = clf.predict(X_test_pca)
    np.save(prefix + '_KNN_pca.npy', y)
    clf = KNeighborsClassifier(n_neighbors = 10)
    clf.fit(X_train_std,y_train)
    y = clf.predict(X_test_std)
    np.save(prefix + '_KNN_std.npy', y)   

    clf = GaussianNB()
    clf.fit(X_train_pca, y_train)
    y = clf.predict(X_test_pca)
    np.save(prefix + '_GaussianNB_pca.npy', y)
    clf = GaussianNB()
    clf.fit(X_train_std, y_train)
    y = clf.predict(X_test_std)
    np.save(prefix + '_GaussianNB_std.npy', y)

    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
    clf.fit(X_train_pca, y_train)
    y = clf.predict(X_test_pca)
    np.save(prefix + '_DT5_pca.npy', y)
    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
    clf.fit(X_train_std, y_train)
    y = clf.predict(X_test_std)
    np.save(prefix + '_DT5_std.npy', y)

    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 2)
    clf.fit(X_train_pca, y_train)
    y = clf.predict(X_test_pca)
    np.save(prefix + '_DT2_pca.npy', y)
    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 2)
    clf.fit(X_train_std, y_train)
    y = clf.predict(X_test_std)
    np.save(prefix + '_DT2_std.npy', y)

    clf = Perceptron(n_iter = 10, penalty = 'l2')
    clf.fit(X_train_pca, y_train)
    y = clf.predict(X_test_pca)
    np.save(prefix + '_Perceptron_pca.npy', y)
    clf = Perceptron(n_iter = 10, penalty = 'l2')
    clf.fit(X_train_std, y_train)
    y = clf.predict(X_test_std)
    np.save(prefix + '_Perceptron_std.npy', y)

    clf = RandomForestClassifier(criterion = 'entropy')
    clf.fit(X_train_pca, y_train)
    y = clf.predict(X_test_pca)
    np.save(prefix + '_RF_pca.npy', y)    
    clf = RandomForestClassifier(criterion = 'entropy')
    clf.fit(X_train_std, y_train)
    y = clf.predict(X_test_std)
    np.save(prefix + '_RF_std.npy', y)  



    #sumsum = 0
    #print y 
    #for i in range(result_row):
    #    sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    #print 'variance of Random Forest Regressor 5 Model is' + str(sumsum/result_row)
'''
    #For rank
    rank_result = {}
    X_train, X_test, y_train, y_test = train_test_split(traindataMatrix, trainMatrix, test_size = 0.2)
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    pca = PCA(n_components = 50)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)




    clf = SVC()
    clf.fit(X_train_pca,y_train)      
    y = clf.predict(X_test_pca)    
    #print y
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['SVM_pca'] = float(sumsum)/float(result_row)

    clf = SVC()
    clf.fit(X_train_std,y_train)      
    y = clf.predict(X_test_std)
    #print y
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['SVM_std'] = float(sumsum)/float(result_row)
    #for i in range(result_row):
    #    sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    #print 'variance of Linear Regression Model is' + str(sumsum/result_row)

    clf = KNeighborsClassifier(n_neighbors = 10)
    clf.fit(X_train_pca,y_train)
    y = clf.predict(X_test_pca)
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['KNN_pca'] = float(sumsum)/float(result_row)


    clf = KNeighborsClassifier(n_neighbors = 10)
    clf.fit(X_train_std,y_train)
    y = clf.predict(X_test_std)
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['KNN_std'] = float(sumsum)/float(result_row)

    clf = GaussianNB()
    clf.fit(X_train_pca, y_train)
    y = clf.predict(X_test_pca)
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['GaussianNB_pca'] = float(sumsum)/float(result_row)
    clf = GaussianNB()
    clf.fit(X_train_std, y_train)
    y = clf.predict(X_test_std)
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['Gaussian_std'] = float(sumsum)/float(result_row)

    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
    clf.fit(X_train_pca, y_train)
    y = clf.predict(X_test_pca)
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['DT5_pca'] = float(sumsum)/float(result_row)
    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
    clf.fit(X_train_std, y_train)
    y = clf.predict(X_test_std)
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['DT5_std'] = float(sumsum)/float(result_row)

    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 2)
    clf.fit(X_train_pca, y_train)
    y = clf.predict(X_test_pca)
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['DT2_pca'] = float(sumsum)/float(result_row)
    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 2)
    clf.fit(X_train_std, y_train)
    y = clf.predict(X_test_std)
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['DT2_std'] = float(sumsum)/float(result_row)

    clf = Perceptron(n_iter = 10, penalty = 'l2')
    clf.fit(X_train_pca, y_train)
    y = clf.predict(X_test_pca)
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['Perceptron_pca'] = float(sumsum)/float(result_row)
    clf = Perceptron(n_iter = 10, penalty = 'l2')
    clf.fit(X_train_std, y_train)
    y = clf.predict(X_test_std)
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['Perceptron_std'] = float(sumsum)/float(result_row)

    clf = RandomForestClassifier(criterion = 'entropy')
    clf.fit(X_train_pca, y_train)
    y = clf.predict(X_test_pca)
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['RF_pca'] = float(sumsum)/float(result_row)   
    clf = RandomForestClassifier(criterion = 'entropy')
    clf.fit(X_train_std, y_train)
    y = clf.predict(X_test_std)
    [result_row] = y.shape    
    sumsum = 0
    for i in range(result_row):
        if(y[i] == y_test[i]):
            sumsum = sumsum + 1
    rank_result['RF_std'] = float(sumsum)/float(result_row)




    rankwrite = open(prefix+'rank2.txt','wb')
    pickle.dump(rank_result, rankwrite)
    rankwrite.close()

    '''
    LogisticModel = LogisticRegression()
    LogisticModel.fit(X_train_std, y_train)
    y = LogisticModel.predict(X_test_std)
    [result_row] = y.shape
    sumsum = 0

    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    print 'variance of Ridge Regression Model is' + str(sumsum)
    '''


    #for i in range(row):
    #    print dataMatrix[i][1]


    '''
    import pandas as pd
    import numpy as np

    def fillMissing(inputcsv, outputcsv):
        
        # read input csv - takes time
        df = pd.read_csv(inputcsv, low_memory=False)
        # Fix date bug
        df.cf4fint = ((pd.to_datetime(df.cf4fint) - pd.to_datetime('1960-01-01')) / np.timedelta64(1, 'D')).astype(int)
        
        # replace NA's with mode
        df = df.fillna(df.mode().iloc[0])
        # if still NA, replace with 1
        df = df.fillna(value=1)
        # replace negative values with 1
        num = df._get_numeric_data()
        num[num < 0] = 1
        # write filled outputcsv
        df.to_csv(outputcsv, index=False)
        
    # Usage:
    fillMissing('background.csv', 'output.csv')
    filleddf = pd.read_csv('output.csv', low_memory=False)
    '''