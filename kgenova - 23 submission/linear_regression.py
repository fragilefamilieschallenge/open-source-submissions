import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from a2core import *

X = joblib.load('train-X.pk')
y = joblib.load('train-y.pk')

gpas = list({x for x in y[:,1]})
gpa_category = range(len(gpas))
gpa_to_cat = {gpas[i]:gpa_category[i] for i in xrange(len(gpas))}
print gpa_to_cat

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=1337)

#for i in range(1,len(y_train[0])):
for i in range(1,2): # Just GPA
    y_cur = list(y_train[:,i])
    #print y_cur
    clf = LassoCV(n_jobs=-1, n_alphas=20)
    clf.fit(X_train,y_cur)
    print "Feature " + str(i)
    y_test_cur = list(y_test[:,i])
    print "Training Accuracy:"
    print clf.score(X_train,y_cur)
    print "Test Accuracy:"
    print clf.score(X_test,y_test_cur)
    print clf.predict(X_test)
