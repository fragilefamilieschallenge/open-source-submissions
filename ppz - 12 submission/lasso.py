# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:57:20 2017

@author: pingping
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
#%%
#read the output.csv file
raw_data=pd.read_csv('output.csv',low_memory=False)

#drop the first column 'idnum'
raw_data=raw_data.drop('idnum',1) 
#where 1 is the axis number (0 for rows and 1 for columns)

#read the train.csv file
train_raw=pd.read_csv('train.csv',low_memory=False)

#pick up those samples from output.csv also appearing in train.csv as x_train
#according to the challlengeid
#http://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
train_data=raw_data.loc[raw_data['challengeID'].isin(train_raw['challengeID'])]

#pick up those samples from output.csv not appearing in train.csv as x_test
test_data=raw_data.loc[~raw_data['challengeID'].isin(train_raw['challengeID'])]
#%%
#sorting the train_data according to challengeid, making the order 
#consistent with the train.csv
train_data=train_data.sort_values(by='challengeID')
####train_data.replace(['Missing'])####
train_data.to_csv('x_train.csv',index=False)
#by setting index=False, the index column in the DataFrame will not be written 
print (train_data.shape)
print (type(train_data))
#drop all columns that have string objects, to note 'string' object, use 'object'
#instead of 'string'. BY doing that, the number of columns is reduced from 
#12944 to 12798
train_data=train_data.select_dtypes(exclude=['object'])
test_data=test_data.select_dtypes(exclude=['object'])
#%%
#read the train.csv file as the y value for the train data
y_train=pd.read_csv('train.csv',low_memory=False)
#to indentify the location of NA values in gpa/girt/materialHardship
#/eviction/layoff/jobTraining columns in train.csv
gpa_nan=pd.isnull(y_train['gpa'])
grit_nan=pd.isnull(y_train['grit'])
materialHardship_nan=pd.isnull(y_train['materialHardship'])
eviction_nan=pd.isnull(y_train['eviction'])
layoff_nan=pd.isnull(y_train['layoff'])
jobTraining_nan=pd.isnull(y_train['jobTraining'])
#%%
#select the y_train data that all gpa/girt/materialHardship
#/eviction/layoff/jobTraining values are NA and thus should be deleted
y_train_gpaNA=y_train.loc[gpa_nan] 
y_train_gritNA=y_train.loc[grit_nan]
y_train_materialHardshipNA=y_train.loc[materialHardship_nan]  
y_train_evictionNA=y_train.loc[eviction_nan]
y_train_layoffNA=y_train.loc[layoff_nan]
y_train_jobTrainingNA=y_train.loc[jobTraining_nan]
#%%
#select the 'gpa' column
y_train_gpa=y_train['gpa']
y_train_grit=y_train['grit']
y_train_materialHardship=y_train['materialHardship']
y_train_eviction=y_train['eviction']
y_train_layoff=y_train['layoff']
y_train_jobTraining=y_train['jobTraining']
#%%
#drop those rows with NA values in gpa/girt/materialHardship
#/eviction/layoff/jobTraining column based on gpa/girt/materialHardship
#/eviction/layoff/jobTraining_nan
y_train_gpa=y_train_gpa.loc[~gpa_nan]
y_train_grit=y_train_grit.loc[~grit_nan]
y_train_materialHardship=y_train_materialHardship.loc[~materialHardship_nan]
y_train_eviction=y_train_eviction.loc[~eviction_nan]
y_train_layoff=y_train_layoff.loc[~layoff_nan]
y_train_jobTraining=y_train_jobTraining.loc[~jobTraining_nan]
#%%
#transfer pandas.core.series.Series to numpy.array
y_gpa=y_train_gpa.values
y_grit=y_train_grit.values
y_materialHardship=y_train_materialHardship.values
y_eviction=y_train_eviction.values
y_layoff=y_train_layoff.values
y_jobTraining=y_train_jobTraining.values
#%%

#Similar to 'train_data=raw_data.loc[raw_data['challengeID'].isin(train_raw['challengeID'])]'
#drop those rows in train data that have NA gpa values in train.csv
train_gpa=train_data.loc[~train_data['challengeID'].isin(y_train_gpaNA['challengeID'])]
train_grit=train_data.loc[~train_data['challengeID'].isin(y_train_gritNA['challengeID'])]
train_materialHardship=train_data.loc[~train_data['challengeID'].isin(y_train_materialHardshipNA['challengeID'])]
train_eviction=train_data.loc[~train_data['challengeID'].isin(y_train_evictionNA['challengeID'])]
train_layoff=train_data.loc[~train_data['challengeID'].isin(y_train_layoffNA['challengeID'])]
train_jobTraining=train_data.loc[~train_data['challengeID'].isin(y_train_jobTrainingNA['challengeID'])]
#%%
#transfer DataFrame to Numpy-array
XtrainGPA=train_gpa.values
XtrainGrit=train_grit.values
XtrainMaterialHard=train_materialHardship.values
XtrainEviction=train_eviction.values
XtrainLayoff=train_layoff.values
XtrainJobTrain=train_jobTraining.values


#%%
para_grid={'alpha': [10, 100, 1000]}
grid=GridSearchCV(Lasso(), para_grid, cv=5)
grid.fit(XtrainGPA, y_gpa)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best alpha value for Lasso:", grid.best_params_)
#%%
value_alpha= [0.001, 0.01, 0.1, 1, 10, 100, 1000]
score_lasso=[0]*7
for i in range(0,7):
    score_lasso[i]=cross_val_score(Lasso(alpha=value_alpha[i]), XtrainGrit, y_grit, cv=5)
    print("Mean cross-validation accuracy of Lasso:{:.2f}".format(np.mean(score_lasso[i])))

#%%

plt.plot(value_alpha, np.mean(score_lasso,axis=1), linewidth=2.0)
plt.xlabel('alpha value for Lasso')
plt.ylabel('Cross-Validation Score')
plt.show()
print (value_alpha)
print (np.mean(score_lasso,axis=1)) #mean of every column(axis=1)
#%%
##cited from 
##https://chrisalbon.com/machine-learning/lasso_regression_in_scikit.html
# Create a function called lasso
def lasso(alphas):
    '''
    Takes in a list of alphas. Outputs a dataframe containing the coefficients of lasso regressions from each alpha.
    '''
    # Create an empty data frame
    df = pd.DataFrame()

    # Create a column of feature names
    df['Feature Name'] = list(train_data) #or train_data.columns.values.tolist() which is faster

    # For each alpha value in the list of alpha values,
    for alpha in alphas:
        # Create a lasso regression with that alpha value,
        lasso = Lasso(fit_intercept=True, alpha=alpha)

        # Fit the lasso regression
        lasso.fit(XtrainGPA, y_gpa)
        
        #predict y_gpa and calculate RMSE for different alpha values
        pred_gpa=lasso.predict(XtrainGPA)
        err = pred_gpa-y_gpa
        total_error = np.dot(err,err)
        rmse_train = np.sqrt(total_error/len(pred_gpa))
    
        # Create a column name for that alpha value
        column_name = 'Alpha = %f' % alpha

        # Create a column of coefficient values
        df[column_name] = lasso.coef_
        
         
    # Return the datafram, df[:10] 
    return df[:10]
#%%
alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000]
for alpha in alphas:
        # Create a lasso regression with that alpha value,
        lasso = Lasso(fit_intercept=True, alpha=alpha)

        # Fit the lasso regression
        lasso.fit(XtrainGPA, y_gpa)
        
        #predict y_gpa and calculate RMSE for different alpha values
        pred_gpa=lasso.predict(XtrainGPA)
        err = pred_gpa-y_gpa
        total_error = np.dot(err,err)
        rmse_train = np.sqrt(total_error/len(pred_gpa))
        # print seven rmse_train values, one per alpha
        print (rmse_train)
#%%
LassoAlpha=lasso([0.001, 0.01, 0.1, 1, 10, 100, 1000])
LassoAlpha.to_csv('LassoAlphas.csv',index=False)

#%%
#let alpha=100 based on the cross-validation above
lasso = Lasso(fit_intercept=True, alpha=100)
lasso.fit(XtrainGPA, y_gpa)
#Predict gpa
pred_gpa=lasso.predict(test_data)
#%%
challengeid=test_data['challengeID'].values


#%%
lasso.fit(XtrainGrit, y_grit)
#Predict grit
pred_grit=lasso.predict(test_data)

#%%
lasso.fit(XtrainMaterialHard, y_materialHardship)
#Predict material hard
pred_materialHard=lasso.predict(test_data)
#%%
pred=np.column_stack((challengeid, pred_gpa, pred_grit, pred_materialHard))

columns=['challengeID','gpa','grit','materialHardship']

GpaGritMaterial=pd.DataFrame(pred, index=range(2121), columns=columns)

#GpaGritMaterial.to_csv('PredictionGGM.csv',index=False)
#%%
