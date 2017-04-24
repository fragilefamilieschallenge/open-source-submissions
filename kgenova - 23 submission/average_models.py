import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import model_selection
from a2core import *
from a2models import *
import os
import sys

def ensemblify(rgr1,rgr2,dim):
    est1 = [r[dim] for r in rgr1]
    est2 = [r[dim] for r in rgr2]
    return [(est1[i] + est2[i])/2.0 for i in xrange(len(est1))]

elasticnet11 = read_csv_to_lists('elasticnet-11-prediction.csv')
elasticnet12 = read_csv_to_lists('elasticnet-12-prediction.csv')
lasso = read_csv_to_lists('lasso-prediction.csv')
multioutput = read_csv_to_lists('multioutput-prediction.csv')

preds = [elasticnet11,elasticnet12,lasso,multioutput]
preds = [p[1:] for p in preds]
preds = [[r[1:] for r in p] for p in preds]
preds = [[[float(v) for v in r] for r in p] for p in preds]
elasticnet11,elasticnet12,lasso,multioutput = preds

y_pred = []
# GPA:
y_pred.append(ensemblify(elasticnet12,multioutput,0))
# Grit:
y_pred.append(ensemblify(elasticnet12,multioutput,1))
# MH:
y_pred.append(ensemblify(elasticnet12,multioutput,2))
# Eviction:
y_pred.append(ensemblify(lasso,multioutput,3))
# Layoff:
y_pred.append(ensemblify(elasticnet11,multioutput,4))
# JT:
y_pred.append(ensemblify(elasticnet12,multioutput,5))

X_challengeIDs = joblib.load('X_challengeIDs.pk')
y_pred_all = [X_challengeIDs] + y_pred

print "Making submission..."
y_names = ['gpa','grit','materialHardship','eviction','layoff','jobTraining']
submission = [['challengeID',y_names[0],y_names[1],y_names[2],y_names[3],y_names[4],y_names[5]]]
submission.extend(transpose(y_pred_all))
subcsv = '\n'.join([','.join([str(x) for x in row]) for row in submission]) + '\n'
f = open('prediction.csv','w')
f.write(subcsv)
f.close()
