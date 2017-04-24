import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import model_selection
from a2core import *
from a2models import *
import os
import sys

regressor = MultiClassElasticnetCVRFitter()
X_train = joblib.load('X_tall.pk').todense()
X_train_name = 'X_tall'
y_train = joblib.load('y_tall.pk')
model_fname = regressor.name() + '-' + X_train_name + '.pk'
if os.path.isfile(model_fname):
    print "Trained model already present. Please fix and rerun."
    sys.exit(1)
regressor.fit(X_train,y_train)
joblib.dump(regressor,model_fname)
