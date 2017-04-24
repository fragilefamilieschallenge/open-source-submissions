'''
-------------------------------------------------------------------
COS 424 - Fundamentals of Machine Learning
Fragile Families Challenge
Authors: Viola Mocz (vmocz) & Sonia Hashim (shashim)


lassoLinearRegression.py
-----------------
Usage: lassoRegression.py -p <path> -f <fileprefix>

Description: Performs lasso regression (L1 penalty) on a given 
  imputation of the Fragile Families data set. Outputs Mean Squared
  Error (MSE) and Coefficient of Deterimination (R2) as evaluation 
  metrics on our training set and outputs predictions in the
  form of a csv file for unlabelled data. Data under - 

  - labeled data                "data/*_outcomeVar_labeled.csv"
  - labels                      "data/*_outcomeVar_train.csv"
  - all data (for prediction)   "data/*_data.csv"        
-------------------------------------------------------------------
'''


### Libraries 
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
import logging 
import numpy as np
import sys
import getopt
import csv
import math

def main(argv):
  ''' Takes in path argument and uses labeled data with median imputation 
      to evaluate the performance of an Lasso Regression (L1 Penalty). Outputs
      Mean Squared Error (MSE) and Coefficient of Deterimination (R2) as 
      evaluation metrics on our training set and outputs predictions in the
      form of a csv file for unlabelled data.'''
  
  ## Path to data, File prefix (fprefix) for imputation
  path = ''
  fprefix = ''

  try:
   opts, args = getopt.getopt(argv,"p:f:",["path=", "fprefix="])
  except getopt.GetoptError:
    print 'Usage: \n python lassoRegression.py -p <path> -f <fprefix>'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print 'Usage: \n python lassoRegression.py -p <path> -f <fprefix>'
      sys.exit()
    elif opt in ("-p", "--path"):
      path = arg
    elif opt in ("-f", "--fprefix"):
      fprefix = arg

  outcome_var = [('gpa', 1), ('grit', 2), ('materialHardship', 3), ('eviction', 4), ('layoff', 5), ('jobTraining', 6)]
  LABEL_INDEX = 0
  NUMERIC_INDEX = 1

  var = [('gpa', 1), ('grit', 2), ('materialHardship', 3)]
  n_var = len(var)

  predictions = np.genfromtxt(path+'/prediction.csv', skip_header = 1, delimiter=',')

  for i in range(n_var):
    v = var[i][LABEL_INDEX]
    var_index = var[i][NUMERIC_INDEX]

    data = np.genfromtxt(path+'/' + fprefix + '_' + v + '_data.csv', skip_header = 1, delimiter = ',')

    ## Fit model using labelled data and print evaluation metrics (MSE, R2)
    x = np.genfromtxt(path + '/' + fprefix + '_' + v +  '_labeled.csv', skip_header = 1, delimiter = ',')
    y = np.genfromtxt(path + '/' + fprefix + '_' + v +  '_train.csv', skip_header = 1, delimiter = ',')
    y = y[:, var_index]

    model = LassoCV(n_alphas = 10)
    model.fit(x, y)

    y_pred = model.predict(x)

    r2 = r2_score(y, y_pred)
    MSE = mean_squared_error(y, y_pred)

    print 'Examining: ' + v
    print 'Using L1-regression, alpha = ', model.alpha_
    print 'Train R2: ', r2
    print 'Train MSE: ', MSE

    ## Find MSE and R2 with K-fold cross validation
    MSE = make_scorer(mean_squared_error)
    r2 = make_scorer(r2_score)
    K = 3

    MSE_scores = cross_val_score(model, x, y, scoring=MSE, cv=K)
    R2_scores = cross_val_score(model, x, y, scoring=r2, cv=K)

    MSE_avg = MSE_scores.mean()
    R2_avg = R2_scores.mean()
    MSE_2std = MSE_scores.std()*2.0
    R2_2std = R2_scores.std()*2.0

    print 'CV MSE Scores: ', MSE_scores
    print 'CV R2 Scores: ', R2_scores
    print 'CV MSE Accuracy: {0:0.3f} +- {1:0.3f}'.format(MSE_avg, MSE_2std)
    print 'CV R2 Accuracy: {0:0.3f} +- {1:0.3f}'.format(R2_avg, R2_2std)


    ## Output predictions into median_prediction.csv 
    pred = model.predict(data)
    predictions[:, var_index] = pred

    ## Get feature names and make a list of the top 50 features
    print 'Top 100 Features: ' 
    with open(path + '/' + fprefix + '_' + v +  '_labeled.csv', 'rb') as f:
      reader = csv.reader(f)
      feat_names  = reader.next()
    c = model.coef_
    a = np.array(map(abs, c))
    top_feat = (a).argsort()
    for j in range(100):
      index = - (j + 1)
      print '{0}. {1}'.format(j+1, feat_names[top_feat[index]])


  names = ','.join(['\"challengeID\"', '\"gpa\"', '\"grit\"', '\"materialHardship\"', '\"eviction\"', '\"layoff\"', '\"jobTraining\"'])
  format = '%i, %1.7f, %1.7f, %1.7f, %1.7f, %1.7f, %1.7f'
  np.savetxt(path+'/lasso_'+ fprefix + '_pred.csv', predictions, delimiter = ',', header=names, fmt = format, comments='')


if __name__ == "__main__":
  main(sys.argv[1:])
