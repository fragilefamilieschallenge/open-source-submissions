import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
## NEWBACKGROUND.CSV GENERATED FROM R SCRIPT ##
df = pd.read_csv("newbackground.csv", low_memory=False)


df = df.sort_values(by = 'challengeID', ascending = 1);

# df = pd.get_dummies(df)
# df = df.as_matrix()

outcomes = pd.read_csv("train.csv", low_memory=False)

# Get rows of training data that have reported gpa
outcomes_gpa = outcomes[pd.notnull(outcomes['gpa'])]
known_gpas = outcomes_gpa.challengeID
print(known_gpas)

# Extract the gpa's
test_outcomes = outcomes_gpa.gpa[1000:1165]
training_outcomes = outcomes_gpa.gpa[0:999]

# Extract rows of raw data
df = df.loc[df['challengeID'].isin(outcomes_gpa.challengeID)]
training_data = df.loc[df['challengeID'].isin(outcomes_gpa.challengeID[0:999])]
test_data = df.loc[df['challengeID'].isin(outcomes_gpa.challengeID[1000:1165])]

# replace NA with NaN
df = df.fillna(np.nan)
# impute replace NA's with mean
# imputer = preprocessing.Imputer(strategy = "most_frequent")
# df = imputer.fit(df)
# training_data = imputer.transform(training_data)
# test_data = imputer.transform(test_data)

print(type(training_data))
# selector = SelectKBest(mutual_info_regression, k=10).fit(training_data, training_outcomes)
# idxs = selector.get_support(indices = True)
# training_data = selector.transform(training_data)
# test_data = test_data[:,idxs]

# # Bootstrap the training set #
clf_rf = RandomForestRegressor()
clf_rf.fit(training_data, training_outcomes)	
print(clf_rf.score(test_data, test_outcomes))
#print(outcomes_gpa.shape)
#print(clf_rf.score(df, outcomes_gpa.gpa))












# print("Done")

## Standardize the data set ##
#scaler = preprocessing.StandardScaler()
#normalized_training_set = scaler.fit_transform(training_set);


