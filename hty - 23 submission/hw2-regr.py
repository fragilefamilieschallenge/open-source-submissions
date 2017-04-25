# %matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

# now using the original X matrix
df = pd.read_csv('Data-SI-small.csv')
train = pd.read_csv('train.csv')

gpa = train['gpa']
grit = train['grit']
mat_hard = train['materialHardship']

idx = df['challengeID']
idy = train['challengeID']


import data_proc_fn as dp
reload(dp)

nan_index_gpa = dp.nan_index(gpa)
nan_index_grit = dp.nan_index(grit)
nan_index_mat_hard = dp.nan_index(mat_hard)


###### these are Y values during fitting
Y_gpa = gpa.drop(gpa.index[nan_index_gpa])
Y_grit = grit.drop(grit.index[nan_index_grit])
Y_mat_hard = mat_hard.drop(mat_hard.index[nan_index_mat_hard])

df_clean = dp.get_clean_X(df)[0]

###### these are X values during fitting
X_gpa = dp.X_with_right_col(df_clean, idy, nan_index_gpa)
X_grit = dp.X_with_right_col(df_clean, idy, nan_index_grit)
X_mat_hard = dp.X_with_right_col(df_clean, idy, nan_index_mat_hard)

print 'now proceed to fitting...'

######## switch estimator here ######

from sklearn.linear_model import ElasticNetCV
estimator = ElasticNetCV(max_iter=10000)

# from sklearn.ensemble import RandomForestRegressor
# estimator = RandomForestRegressor()

import regressor as regr
reload(regr)

gpa_out = regr.processing(estimator, df_clean, X_gpa, Y_gpa)
grit_out = regr.processing(estimator, df_clean, X_grit, Y_grit)
mat_hard_out = regr.processing(estimator, df_clean, X_mat_hard, Y_mat_hard)


######### writing outputs
prediction = pd.read_csv('prediction.csv')

prediction['gpa'] = gpa_out
prediction['grit'] = grit_out
prediction['materialHardship'] = mat_hard_out

prediction.to_csv('prediction1.csv', index=False)
