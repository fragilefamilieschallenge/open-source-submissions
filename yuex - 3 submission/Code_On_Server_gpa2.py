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
from sklearn.metrics import r2_score
from time import clock
import pdb
import pickle

prefix_list = ['Processing_Missing_Data_By_Just_Mean.txt', 'Processing_Missing_Data_By_Just_Mean_Threshold02.txt', 'Processing_Missing_Data_By_Just_Mean_Threshold06.txt', 'Whole_Dataset_Processing_Missing_Point','Whole_Dataset_Processing_Missing_Point_02Threshold', 'Whole_Dataset_Processing_Missing_Point_06Threshold']
for pfi in range(6):
    prefix = prefix_list[pfi]
    testclass = 'gpa'
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
            float(gpa_item)
            trainMatrix[i] = float(gpa_item)
            trainTag.append(int(train_data[i]['challengeID']))
    #        useful_rows.append(float(trainTag[i]))
        except ValueError:
            pass

    trainMatrix = trainMatrix[trainMatrix>=0]
    for i in range(len(trainTag)):
        trainTag[i] = trainTag[i] - 1

    dataMatrix = np.load(prefix+'.npy')
    traindataMatrix = dataMatrix[trainTag]

    prefix = prefix_list[pfi] + 'gpa'
    stdsc = StandardScaler()

    rank_result = {}
    rs_score = {}
    X_train, X_test, y_train, y_test = train_test_split(traindataMatrix, trainMatrix, test_size = 0.2)
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    pca = PCA(n_components = 50)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)


    lrModel = LinearRegression()
    lrModel.fit(X_train_pca, y_train)
    y = lrModel.predict(X_test_pca)
    [result_row] = y.shape
    #print y
    sumsum = 0
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['lr_pca'] = sumsum/float(result_row)
    rs_score['lr_pca'] = r2_score(y_test, y)    
    lrModel = LinearRegression()
    lrModel.fit(X_train_std, y_train)
    y = lrModel.predict(X_test_std)
    [result_row] = y.shape
    #print y
    sumsum = 0
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['lr_std'] = sumsum/float(result_row)
    rs_score['lr_std'] = r2_score(y_test, y)
    ridgeModel = Ridge()
    ridgeModel.fit(X_train_pca, y_train)
    y = ridgeModel.predict(X_test_pca)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    #print 'variance of Ridge Regression Model is' + str(sumsum/result_row)
    rank_result['ridge_pca'] = sumsum/float(result_row)
    rs_score['ridge_pca'] = r2_score(y_test, y)    
    ridgeModel = Ridge()
    ridgeModel.fit(X_train_std, y_train)
    y = ridgeModel.predict(X_test_std)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    #print 'variance of Ridge Regression Model is' + str(sumsum/result_row)
    rank_result['ridge_std'] = sumsum/float(result_row)
    rs_score['ridge_std'] = r2_score(y_test, y)
    lassoModel = Lasso()
    lassoModel.fit(X_train_pca, y_train)
    y = lassoModel.predict(X_test_pca)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['lasso_pca'] = sumsum/float(result_row)
    rs_score['lasso_pca'] = r2_score(y_test, y)    
    lassoModel = Lasso()
    lassoModel.fit(X_train_std, y_train)
    y = lassoModel.predict(X_test_std)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['lasso_std'] = sumsum/float(result_row)
    rs_score['lasso_std'] = r2_score(y_test, y)
    ElasticModel = ElasticNetCV()
    ElasticModel.fit(X_train_pca, y_train)
    y = ElasticModel.predict(X_test_pca)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['Elastic_pca'] = sumsum/float(result_row)
    rs_score['Elastic_pca'] = r2_score(y_test, y)    
    ElasticModel = ElasticNetCV()
    ElasticModel.fit(X_train_std, y_train)
    y = ElasticModel.predict(X_test_std)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['Elastic_std'] = sumsum/float(result_row)
    rs_score['Elastic_std'] = r2_score(y_test, y)
    LarsModel = Lars()
    LarsModel.fit(X_train_pca, y_train)
    y = LarsModel.predict(X_test_pca)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['Lars_pca'] = sumsum/float(result_row)
    rs_score['Lars_pca'] = r2_score(y_test, y)    
    LarsModel = Lars()
    LarsModel.fit(X_train_std, y_train)
    y = LarsModel.predict(X_test_std)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['Lars_std'] = sumsum/float(result_row)
    rs_score['Lars_std'] = r2_score(y_test, y)

    LassoLarsModel = LassoLars()
    LassoLarsModel.fit(X_train_pca, y_train)
    y = LassoLarsModel.predict(X_test_pca)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['LassoLars_pca'] = sumsum/float(result_row)
    rs_score['LassoLars_pca'] = r2_score(y_test, y)    
    LassoLarsModel = LassoLars()
    LassoLarsModel.fit(X_train_std, y_train)
    y = LassoLarsModel.predict(X_test_std)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['LassoLars_std'] = sumsum/float(result_row)
    rs_score['LassoLars_std'] = r2_score(y_test, y)
    ompModel = OrthogonalMatchingPursuit()
    ompModel.fit(X_train_pca, y_train)
    y = ompModel.predict(X_test_pca)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['OM_pca'] = sumsum/float(result_row)
    rs_score['OM_pca'] = r2_score(y_test, y)    
    ompModel = OrthogonalMatchingPursuit()
    ompModel.fit(X_train_std, y_train)
    y = ompModel.predict(X_test_std)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['OM_std'] = sumsum/float(result_row)
    rs_score['OM_std'] = r2_score(y_test, y)
    BRModel = BayesianRidge()
    BRModel.fit(X_train_pca, y_train)
    y = BRModel.predict(X_test_pca)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['BR_pca'] = sumsum/float(result_row)
    rs_score['BR_pca'] = r2_score(y_test, y)
    BRModel = BayesianRidge()
    BRModel.fit(X_train_std, y_train)
    y = BRModel.predict(X_test_std)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['BR_std'] = sumsum/float(result_row)
    rs_score['BR_std'] = r2_score(y_test, y)
    ARDModel = ARDRegression()
    ARDModel.fit(X_train_pca, y_train)
    y = ARDModel.predict(X_test_pca)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['ARD_pca'] =sumsum/float(result_row)
    rs_score['ARD_pca'] = r2_score(y_test, y)    
    ARDModel = ARDRegression()
    ARDModel.fit(X_train_std, y_train)
    y = ARDModel.predict(X_test_std)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['ARD_std'] =sumsum/float(result_row)
    rs_score['ARD_std'] = r2_score(y_test, y)


    DTRModel = DecisionTreeRegressor(max_depth = 2)
    DTRModel.fit(X_train_pca, y_train)
    y = DTRModel.predict(X_test_pca)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['DTR2_pca'] = sumsum/float(result_row)
    rs_score['DTR2_pca'] = r2_score(y_test, y)    
    DTRModel = DecisionTreeRegressor(max_depth = 2)
    DTRModel.fit(X_train_std, y_train)
    y = DTRModel.predict(X_test_std)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['DTR2_std'] = sumsum/float(result_row)
    rs_score['DTR2_std'] = r2_score(y_test, y)


    DTRModel = DecisionTreeRegressor(max_depth = 5)
    DTRModel.fit(X_train_pca, y_train)
    y = DTRModel.predict(X_test_pca)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['DTR5_pca'] = sumsum/float(result_row)
    rs_score['DTR5_pca'] = r2_score(y_test, y)    
    DTRModel = DecisionTreeRegressor(max_depth = 5)
    DTRModel.fit(X_train_std, y_train)
    y = DTRModel.predict(X_test_std)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['DTR5_std'] = sumsum/float(result_row)
    rs_score['DTR5_std'] = r2_score(y_test, y)

    RFRModel = RandomForestRegressor()
    RFRModel.fit(X_train_pca, y_train)
    y = RFRModel.predict(X_test_pca)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['RFR_pca'] = sumsum/float(result_row)
    rs_score['RFR_pca'] = r2_score(y_test, y)    
    RFRModel = RandomForestRegressor()
    RFRModel.fit(X_train_std, y_train)
    y = RFRModel.predict(X_test_std)
    [result_row] = y.shape
    sumsum = 0
    #print y 
    for i in range(result_row):
        sumsum = sumsum + (y[i]-y_test[i]) *(y[i] - y_test[i])
    rank_result['RFR_std'] = sumsum/float(result_row)
    rs_score['RFR_std'] = r2_score(y_test, y)
#    rankwrite = open(prefix+'rank2.txt','wb')
#    pickle.dump(rank_result, rankwrite)
#    rankwrite.close()
    rswrite = open(prefix+'rs2.txt','wb')
    pickle.dump(rs_score, rswrite)
    rswrite.close()
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