import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

## NEWBACKGROUND.CSV GENERATED FROM R SCRIPT ##
df = pd.read_csv("newbackground.csv", low_memory=False)


df = df.sort_values(by = 'challengeID', ascending = 1);




# df = pd.get_dummies(df)
# df = df.as_matrix()

outcomes = pd.read_csv("train.csv", low_memory=False)

outcomes_gpa = outcomes[pd.notnull(outcomes['gpa'])]
known_gpas = outcomes_gpa.challengeID
print(known_gpas)

df = df.loc[df['challengeID'].isin(outcomes_gpa.challengeID)]


# replace NA with NaN
df = df.fillna(np.nan)
# impute replace NA's with mean
imputer = preprocessing.Imputer()
df = imputer.fit_transform(df)

# # Bootstrap the training set #
clf_rf = RandomForestRegressor()
clf_rf.fit(df, outcomes_gpa.gpa)
print(outcomes_gpa.shape)
print(clf_rf.score(df, outcomes_gpa.gpa))






# print("Done")

## Standardize the data set ##
#scaler = preprocessing.StandardScaler()
#normalized_training_set = scaler.fit_transform(training_set);


