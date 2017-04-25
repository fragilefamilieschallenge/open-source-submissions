%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np


df = pd.read_csv('background.csv')
train = pd.read_csv('train.csv')

eviction = train['eviction']
layoff = train['layoff']
job = train['jobTraining']

idx = df['challengeID']
idy = train['challengeID']


import data_proc_fn as dp
reload(dp)

nan_index_eviction = dp.nan_index(eviction)
nan_index_layoff = dp.nan_index(layoff)
nan_index_job = dp.nan_index(job)

###### these are Y values during fitting
Y_evic = eviction.drop(eviction.index[nan_index_eviction])

Y_layoff = layoff.drop(layoff.index[nan_index_layoff])

Y_job = job.drop(job.index[nan_index_job])

df_clean = dp.get_clean_X(df)[0]

###### these are X values during fitting
X_evic = dp.X_with_right_col(df_clean, idy, nan_index_eviction)
X_layoff = dp.X_with_right_col(df_clean, idy, nan_index_layoff)
X_job = dp.X_with_right_col(df_clean, idy, nan_index_job)

####### try random forest
from sklearn.ensemble import RandomForestClassifier
estimator = RandomForestClassifier()

# # initialize logistic regressor
# estimator = linear_model.LogisticRegression(n_jobs=-1)

import classifier as clf
reload(clf)

evic_out = clf.processing(estimator, df_clean, X_evic, Y_evic.astype(bool))
layoff_out = clf.processing(estimator, df_clean, X_layoff, Y_layoff.astype(bool))
job_out = clf.processing(estimator, df_clean, X_job, Y_job.astype(bool))


##### find important features
temp= estimator.feature_importances_

# list important features in descending order
temp_ind = (-temp).argsort()[0:20]

# find out the corresponding column
df.columns[temp_ind]
