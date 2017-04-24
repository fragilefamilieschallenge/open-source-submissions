# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 10:58:14 2017

@author: pingping
"""
from sklearn.linear_model import SGDRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
scaler = StandardScaler()
scaler.fit(XtrainGPA)
x_gpa = scaler.transform(XtrainGPA)
test = scaler.transform(test_data)
#%%
value_alpha= [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
score_sgd=[0]*9
for i in range(0,9):
    score_sgd[i]=cross_val_score(SGDRegressor(penalty='l2', alpha=value_alpha[i], n_iter=200), x_gpa, y_gpa, cv=5)
    #print("Mean cross-validation accuracy of SGD:{:.2f}".format(np.mean(score_sgd[i]])))
#%%
plt.semilogx(value_alpha, np.mean(score_sgd,axis=1), linewidth=2.0)
plt.xlabel('alpha value for SGD')
plt.ylabel('Cross-Validation Score')
plt.show()
print (value_alpha)
print (np.mean(score_sgd,axis=1)) #mean of every column(axis=1)
#%%
para_grid={'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]}
grid=GridSearchCV(SGDRegressor(), para_grid, cv=5)
grid.fit(x_gpa, y_gpa)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best alpha value for SGD:", grid.best_params_)
#%%
def sgd_regressor(alphas):
    '''
    Takes in a list of alphas. Outputs a dataframe containing the coefficients of SGDRegressor regressions from each alpha.
    '''
    # Create an empty data frame
    df = pd.DataFrame()

    # Create a column of feature names
    df['Feature Name'] = list(train_data) #or train_data.columns.values.tolist() which is faster

    # For each alpha value in the list of alpha values,
    for alpha in alphas:
        # Create a lasso regression with that alpha value,
        sgd_regressor = SGDRegressor(penalty='l2', fit_intercept=True, alpha=alpha, n_iter=200)

        # Fit the lasso regression
        sgd_regressor.fit(x_gpa, y_gpa)

        # Create a column name for that alpha value
        column_name = 'Alpha = %f' % alpha

        # Create a column of coefficient values
        df[column_name] = sgd_regressor.coef_

    # Return the datafram    
    return df[:10]

#%%
alphas= [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
for alpha in alphas:
        # Create a lasso regression with that alpha value,
        sgd = SGDRegressor(penalty='l2', fit_intercept=True, alpha=alpha, n_iter=200)

        # Fit the lasso regression
        sgd.fit(x_gpa, y_gpa)
        
        #predict y_gpa and calculate RMSE for different alpha values
        pred_gpa=sgd.predict(x_gpa)
        err = pred_gpa-y_gpa
        total_error = np.dot(err,err)
        rmse_train = np.sqrt(total_error/len(pred_gpa))
        # print seven rmse_train values, one per alpha
        print (rmse_train)
        
#%%
SgdAlpha=sgd_regressor([ 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000])
SgdAlpha.to_csv('SgdAlphas.csv',index=False)
#%%
#let alpha=1000 based on the cross-validation above
sgd = SGDRegressor(penalty='l2',fit_intercept=True, alpha=1000, n_iter=200)
sgd.fit(x_gpa, y_gpa)
#Predict gpa
pred_gpa=sgd.predict(test)

#%%
scaler.fit(XtrainGrit)
x_grit = scaler.transform(XtrainGrit)

sgd = SGDRegressor(penalty='l2',fit_intercept=True, alpha=1000, n_iter=200)
sgd.fit(x_grit, y_grit)
#Predict gpa
pred_grit=sgd.predict(test)

#%%
scaler.fit(XtrainMaterialHard)
x_materialHard = scaler.transform(XtrainMaterialHard)

sgd = SGDRegressor(penalty='l2',fit_intercept=True, alpha=1000, n_iter=200)
sgd.fit(x_materialHard, y_materialHardship)
#Predict gpa
pred_materialHard=sgd.predict(test)
#%%
pred=np.column_stack((challengeid, pred_gpa, pred_grit, pred_materialHard, pred_eviction, pred_layoff, pred_jobtraining))
columns=['challengeID','gpa','grit','materialHardship', 'eviction', 'layoff', 'jobTraining']
prediction=pd.DataFrame(pred, index=range(2121), columns=columns)
prediction.to_csv('prediction.csv',index=False)