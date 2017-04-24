'''
-------------------------------------------------------------------
COS 424 - Fundamentals of Machine Learning
Fragile Families Challenge
Authors: Viola Mocz (vmocz) & Sonia Hashim (shashim)


LinRegMult.py
-----------------
Usage: LinRegMult.py -p <path>

Description: Performs ordinary linear regression on the multiple
  imputation of the Fragile Families data set. Outputs predictions
  in the form of a csv file for unlabelled data.Data under -

  - labeled data                "data/amelia*_labeled.csv"
  - labels                      "data/amelia*_train.csv"
  - all data (for prediction)   "data/amelia*_data.csv"
-------------------------------------------------------------------
'''


### Libraries 
from sklearn import linear_model
import numpy as np
import sys
import getopt

def main(argv):
  ''' Takes in path argument and uses labeled data with multiple imputation
      to evaluate the performance of an Ordinary Linear Regression. Outputs
      predictions in the form of a csv file for unlabelled data.'''

  ## Path to data (note specific file names given above)
  path = ''

  try:
    opts, args = getopt.getopt(argv, "p:", ["path="])
  except getopt.GetoptError:
    print 'Usage: \n python LinRegMult.py -p <path>'
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print 'Usage: \n python LinRegMult.py -p <path>'
      sys.exit()
    elif opt in ("-p", "--path"):
      path = arg

  ## Fit model using labelled data
  x1 = np.genfromtxt(path + '/amelia1_labeled.csv', skip_header=1, delimiter=',')
  x2 = np.genfromtxt(path + '/amelia2_labeled.csv', skip_header=1, delimiter=',')
  x3 = np.genfromtxt(path + '/amelia3_labeled.csv', skip_header=1, delimiter=',')
  x4 = np.genfromtxt(path + '/amelia4_labeled.csv', skip_header=1, delimiter=',')
  x5 = np.genfromtxt(path + '/amelia5_labeled.csv', skip_header=1, delimiter=',')
  x6 = np.genfromtxt(path + '/amelia6_labeled.csv', skip_header=1, delimiter=',')
  x7 = np.genfromtxt(path + '/amelia7_labeled.csv', skip_header=1, delimiter=',')
  x8 = np.genfromtxt(path + '/amelia8_labeled.csv', skip_header=1, delimiter=',')
  x9 = np.genfromtxt(path + '/amelia9_labeled.csv', skip_header=1, delimiter=',')
  x10 = np.genfromtxt(path + '/amelia10_labeled.csv', skip_header=1, delimiter=',')

  y = np.genfromtxt(path + '/amelia1_train.csv', skip_header=1, delimiter=',')

  GRIT_INDEX = 2
  y = y[:, GRIT_INDEX]

  model1 = linear_model.LinearRegression()
  model2 = linear_model.LinearRegression()
  model3 = linear_model.LinearRegression()
  model4 = linear_model.LinearRegression()
  model5 = linear_model.LinearRegression()
  model6 = linear_model.LinearRegression()
  model7 = linear_model.LinearRegression()
  model8 = linear_model.LinearRegression()
  model9 = linear_model.LinearRegression()
  model10 = linear_model.LinearRegression()

  model1.fit(x1, y)
  model2.fit(x2, y)
  model3.fit(x3, y)
  model4.fit(x4, y)
  model5.fit(x5, y)
  model6.fit(x6, y)
  model7.fit(x7, y)
  model8.fit(x8, y)
  model9.fit(x9, y)
  model10.fit(x10, y)

  ## Output predictions into amelia_prediction.csv
  predictions = np.genfromtxt(path + '/prediction.csv', skip_header=1, delimiter=',')
  data = np.genfromtxt(path + '/amelia1_data.csv', skip_header=1, delimiter=',')
  grit_pred1 = model1.predict(data)
  grit_pred2 = model2.predict(data)
  grit_pred3 = model3.predict(data)
  grit_pred4 = model4.predict(data)
  grit_pred5 = model5.predict(data)
  grit_pred6 = model6.predict(data)
  grit_pred7 = model7.predict(data)
  grit_pred8 = model8.predict(data)
  grit_pred9 = model9.predict(data)
  grit_pred10 = model10.predict(data)

  grit_pred = np.mean(np.array([grit_pred1,grit_pred2,grit_pred3,grit_pred4,grit_pred5,grit_pred6,grit_pred7,grit_pred8,grit_pred9,grit_pred10]), axis=0)


  predictions[:, GRIT_INDEX] = grit_pred
  names = ','.join(['\"challengeID\"', '\"gpa\"', '\"grit\"', '\"materialHardship\"', '\"eviction\"', '\"layoff\"', '\"jobTraining\"'])
  format = '%i, %1.7f, %1.7f, %1.7f, %1.7f, %1.7f, %1.7f'
  np.savetxt(path+'/amelia10_pred.csv', predictions, delimiter = ',', header=names, fmt = format, comments='')


if __name__ == "__main__":
  main(sys.argv[1:])
