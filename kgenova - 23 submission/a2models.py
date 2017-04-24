from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import svm
from sklearn import model_selection
from sklearn import tree
from sklearn import naive_bayes
from sklearn import pipeline
from sklearn import gaussian_process
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import ensemble
import time
import a2core
import numpy as np

class BayesianRidgeRegressionFitter:
    def fit(self,X,y):
        print "Fitting a bayesian ridge regressor..."
        self.standardizer = preprocessing.StandardScaler()
        X = self.standardizer.fit_transform(X)
        self.clf = linear_model.BayesianRidge()
        self.clf.fit(X,y)

    def predict(self,X):
        X = self.standardizer.transform(X)
        return self.clf.predict(X)

    def name(self):
        return 'bayesian-ridge'

class LinearRegressionFitter:
    def fit(self,X,y):
        print "Fitting a LinearRegression regressor..."
        self.standardizer = preprocessing.StandardScaler()
        X = self.standardizer.fit_transform(X)
        self.clf = linear_model.LinearRegression(n_jobs=12,normalize=True)
        self.clf.fit(X,y)

    def predict(self,X):
        X = self.standardizer.transform(X)
        return self.clf.predict(X)

    def name(self):
        return 'linear-regression-regressor'

class LassoRegressionFitter:
    def fit(self,X,y):
        print "Fitting a LassoCV regressor..."
        self.standardizer = preprocessing.StandardScaler()
        X = self.standardizer.fit_transform(X)
        cv = model_selection.ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        self.clf = linear_model.LassoCV(n_alphas=100,cv=cv,n_jobs=7,normalize=[True,False])
        self.clf.fit(X,y)

    def predict(self,X):
        X = self.standardizer.transform(X)
        return self.clf.predict(X)

    def name(self):
        return 'lasso-cv-regressor'

class RidgeRegressionFitter:
    def fit(self,X,y):
        print "Fitting a RidgeCV regressor..."
        self.standardizer = preprocessing.StandardScaler()
        X = self.standardizer.fit_transform(X)
        cv = model_selection.ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        self.clf = linear_model.RidgeCV(alphas=[0.01,0.1,1.,10.],cv=cv,normalize=[True,False])
        self.clf.fit(X,y)

    def predict(self,X):
        X = self.standardizer.transform(X)
        return self.clf.predict(X)

    def name(self):
        return 'ridge-cv-regressor'

class ElasticNetRegressionFitter:
    def fit(self,X,y):
        print "Fitting a restricted ElasticNetCV regressor..."
        self.standardizer = preprocessing.StandardScaler()
        X = self.standardizer.fit_transform(X)
        cv = model_selection.ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        alpha_range = [0.005, 0.007, 0.002, 0.0025, 0.004, 0.003, 0.0035877427142009029,0.01,0.001]
        param_grid = []
        param_grid.append(dict(alpha=alpha_range,l1_ratio=[.1,.2,.25,.3,.35,.4,.5,.6,.65,.7,.8],normalize=[True],max_iter=[10000]))
        print "Using param grid " + str(param_grid)
        self.clf = model_selection.GridSearchCV(linear_model.ElasticNet(),param_grid=param_grid,cv=cv,n_jobs=12)
        self.clf.fit(X,y)
        print "Best params: " + str(self.clf.best_params_) + " and corresponding score is " + str(self.clf.best_score_)

    def predict(self,X):
        X = self.standardizer.transform(X)
        return self.clf.predict(X)

    def name(self):
        return 'elasticnet-regressor'

class ElasticNetCVRegressionFitter:
    def fit(self,X,y):
        print "Fitting an ElasticNetCV regressor..."
        self.standardizer = preprocessing.StandardScaler()
        X = self.standardizer.fit_transform(X)
        cv = model_selection.ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        self.clf = linear_model.ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],cv=cv,n_jobs=7,normalize=True)
        self.clf.fit(X,y)

    def predict(self,X):
        X = self.standardizer.transform(X)
        return self.clf.predict(X)

    def name(self):
        return 'elasticnet-cv-regressor'

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y):
        return self
    def transform(self, X):
        return X[:,:-1096]

class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y):
        return self
    def transform(self, X):
        return X[:,-1096:]

class EnsembleNBFitter:
    def fit(self,X,y):
        # Split into categorical,numerical categories:
        self.cat_clf = pipeline.Pipeline((('cat-tf',CategoricalTransformer()),('bnb',naive_bayes.BernoulliNB())))
        self.num_clf = pipeline.Pipeline((('num-tf',NumericalTransformer()),('scaler',preprocessing.StandardScaler()),('gnb',naive_bayes.GaussianNB())))
        self.clf = ensemble.VotingClassifier(estimators=[('num-clf',self.num_clf),('cat-clf',self.cat_clf)])
        self.clf.fit(X,y)

    def predict(self,X):
        return self.clf.predict(X)
    def decision(self,X):
        return None
    def name(self):
        return 'split-nb-even-clf'

class EnsembleNBCVFitter:
    def fit(self,X,y):
        # Split into categorical,numerical categories:
        self.cat_clf = pipeline.Pipeline((('cat-tf',CategoricalTransformer()),('bnb',naive_bayes.BernoulliNB())))
        self.num_clf = pipeline.Pipeline((('num-tf',NumericalTransformer()),('gnb',naive_bayes.GaussianNB())))
        weights_range=[[a,1.0-a] for a in [0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]]
        voting_range= ['soft']
        param_grid = dict(voting=voting_range,weights=weights_range)
        print "Using param grid " + str(param_grid)
        cv = model_selection.StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        self.clf = ensemble.VotingClassifier(estimators=[('num-clf',self.num_clf),('cat-clf',self.cat_clf)])
        self.clf = model_selection.GridSearchCV(self.clf,param_grid=param_grid,cv=cv,n_jobs=7)
        self.clf.fit(X,y)
        print "Best params: " + str(self.clf.best_params_) + " and corresponding score is " + str(self.clf.best_score_)

    def predict(self,X):
        return self.clf.predict(X)
    def decision(self,X):
        return None
    def name(self):
        return 'split-nb-clf'

class MultiClassElasticnetCVRFitter:
    def fit(self,X,y):
        print "Fitting a multiclass ElasticNet regressor..."
        assert(y.shape[1] == 6)

        self.standardizer = preprocessing.StandardScaler()
        X = self.standardizer.fit_transform(X)
        cv = model_selection.ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)

        alpha_range = [0.005, 0.007, 0.002, 0.0025, 0.004, 0.003, 0.0035877427142009029,0.01,0.001]
        param_grid = []
        param_grid.append(dict(alpha=alpha_range,l1_ratio=[.1,.2,.25,.3,.35,.4,.5,.6,.65,.7,.8],normalize=[True],max_iter=[10000]))
        print "Using param grid " + str(param_grid)
        self.clf = model_selection.GridSearchCV(linear_model.MultiTaskElasticNet(),param_grid=param_grid,cv=cv,n_jobs=12)
        self.clf.fit(X,y)
        print "Best params: " + str(self.clf.best_params_) + " and corresponding score is " + str(self.clf.best_score_)

    def predict(self,X):
        X = self.standardizer.transform(X)
        return self.clf.predict(X)

    def name(self):
        return 'multitask-elasticnet-6-regressor'

class MultiTaskRegressionAdapter:
    def __init__(self,clf,dimension):
        self.dim = dimension
        self.clf = clf
        #self.model = fitted_multitask_regressor

    def fit(self,X,y):
        assert(False)

    def predict(self,X):
        return transpose(self.clf.predict(X))[self.dim]

    def name(self):
        return self.clf.name() + '-regression-adapter'


class LogisticRegressionFitter:
    def fit(self,X,y):
        print "Fitting a Logistic Regression model..."
        self.standardizer = preprocessing.StandardScaler()
        X = self.standardizer.fit_transform(X)
        self.clf = linear_model.LogisticRegressionCV(n_jobs=7)
        self.clf.fit(X,y)

    def predict(self,X):
        X = self.standardizer.transform(X)
        return self.clf.predict(X)

    def decision(self,X):
        X = self.standardizer.transform(X)
        return self.clf.decision_function(X)

    def name(self):
        return 'logistic-regression'

class GaussianProcessClassifierFitter:
    def fit(self,X,y):
        print "Fitting a Gaussian Process Classification model..."
        self.standardizer = preprocessing.StandardScaler()
        X = self.standardizer.fit_transform(X)
        self.clf = gaussian_process.GaussianProcessClassifier()
        self.clf.fit(X,y)

    def predict(self,X):
        X = self.standardizer.transform(X)
        return self.clf.predict(X)

    def decision(self,X):
        return None

    def name(self):
        return 'gaussian-process-classifier'

class SGDClassifierFitter:
    def fit(self,X,y):
        print "Fitting an SGD Elasticnet Classification model..."
        t_start = time.time()
        n_iter = np.ceil(10**6 / float(len(y)))

        self.standardizer = preprocessing.StandardScaler()
        X = self.standardizer.fit_transform(X)
        self.clf = linear_model.SGDClassifier(loss='modified_huber',penalty='elasticnet',random_state=1337,n_jobs=7,n_iter=n_iter)
        self.clf.fit(X,y)
        utime = time.time() - t_start
        print " Done fitting. Took time " + str(utime)

    def predict(self,X):
        X = self.standardizer.transform(X)
        return self.clf.predict(X)

    def decision(self,X):
        return None

    def name(self):
        return 'sgd-classifier'

class SGDClassifierCVFitter:
    def fit(self,X,y):
        print "Fitting an SGD Elasticnet Classification model..."
        t_start = time.time()
        n_iter = np.ceil(10**6 / float(len(y)))

        self.standardizer = preprocessing.StandardScaler()
        X = self.standardizer.fit_transform(X)

        alpha_range = 10.0**-np.arange(1,7)
        param_grid = []
        param_grid.append(dict(loss=['log','modified_huber'],alpha=alpha_range,n_iter=[n_iter],penalty=['elasticnet'],l1_ratio=[.1, .5, .7, .9, .95, .99, 1.]))
        print "Using param grid " + str(param_grid)
        self.clf = linear_model.SGDClassifier(random_state=1337)
        cv = model_selection.StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        self.clf = model_selection.GridSearchCV(self.clf,param_grid=param_grid,cv=cv,n_jobs=7)
        self.clf.fit(X,y)
        print "Best params: " + str(self.clf.best_params_) + " and corresponding score is " + str(self.clf.best_score_)

        utime = time.time() - t_start
        print " Done fitting. Took time " + str(utime)

    def predict(self,X):
        X = self.standardizer.transform(X)
        return self.clf.predict(X)

    def decision(self,X):
        return None

    def name(self):
        return 'sgd-classifier-cv'

class NonlinearSVCGridSearchFitter:
    def fit(self,X,y):
        print "Fitting a GridCV RBF SVM..."
        self.standardizer = preprocessing.StandardScaler()
        X = self.standardizer.fit_transform(X)
        self.selector = feature_selection.SelectKBest(feature_selection.f_classif, k=1000)
        X = self.selector.fit_transform(X,y)
        C_range = [2.0**i for i in [-5,-3,-1,1,3,5,7,9,11,13,15]]
        gamma_range = [2.0**i for i in  [-15,-13,-11,-9,-7,-5,-3,-1,1,3]]
        param_grid = dict(C=C_range,gamma=gamma_range)
        print "Using param grid " + str(param_grid)
        cv = model_selection.StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        self.clf = model_selection.GridSearchCV(svm.SVC(),param_grid=param_grid,cv=cv,n_jobs=7)
        self.clf.fit(X,y)
        print "Best params: " + str(self.clf.best_params_) + " and corresponding score is " + str(self.clf.best_score_)

    def predict(self,X):
        X = self.standardizer.transform(X)
        X = self.selector.transform(X)
        return self.clf.predict(X)

    def decision(self,X):
        X = self.standardizer.transform(X)
        X = self.selector.transform(X)
        return self.clf.decision_function(X)

    def name(self):
        return 'sss-gridcv-svc'

class DecisionTreeClassifierFitter:
    def fit(self,X,y):
        print "Fitting a Decision Tree Classifier..."
        self.clf = tree.DecisionTreeClassifier(random_state=1337)
        self.clf.fit(X,y)

    def predict(self,X):
        return self.clf.predict(X)

    def decision(self,X):
        return None

    def name(self):
        return 'dtc'
