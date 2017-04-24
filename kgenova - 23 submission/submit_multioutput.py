import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import model_selection
from a2core import *
from a2models import *
import os
import sys

regressor = MultiClassElasticnetCVRFitter()
X_full = joblib.load('X.pk').todense()
model_fname = regressor.name() + '-X_tall.pk'
if not os.path.isfile(model_fname):
    print "Trained model not present. Please fix and rerun."
    sys.exit(1)
regressor = joblib.load(model_fname)
print "Predicting..."
y_pred = regressor.predict(X_full)

X_challengeIDs = joblib.load('X_challengeIDs.pk')
y_pred_all = [X_challengeIDs] + transpose(y_pred.tolist())

print "Making submission..."
y_names = ['gpa','grit','materialHardship','eviction','layoff','jobTraining']
submission = [['challengeID',y_names[0],y_names[1],y_names[2],y_names[3],y_names[4],y_names[5]]]
submission.extend(transpose(y_pred_all))
subcsv = '\n'.join([','.join([str(x) for x in row]) for row in submission]) + '\n'
f = open('prediction.csv','w')
f.write(subcsv)
f.close()
