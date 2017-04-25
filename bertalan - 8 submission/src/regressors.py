'''
Created on Apr 2, 2017

@author: tsbertalan
'''
import numpy as np

from sklearn import svm
from sklearn.linear_model import MultiTaskElasticNet, Ridge, Lasso

import keras.wrappers.scikit_learn
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input

from utilities import timeMethodDecorator
import tqdm


class KerasRegressor(object):
    
    def __init__(self,
                 ninputs, noutputs,
                 nlayers=1, layerSizes=[60],
                 dropoutFrac=0.5,
                 compileKwargs=dict(
                     loss='binary_crossentropy', optimizer='adam',
                     metrics=['mse']
                 ),
                ):

        def makeModel():
#             # Functional API
#             x = inputTensor = Input(shape=(ninputs,))
#             x = Dense(
#                 layerSizes[0],
#                 kernel_initializer='normal', activation='relu',
#             )(x)
#             
#             # Add additional layers if requested.
#             for i in range(nlayers-1):
#                 x = Dense(layerSizes[i+1], kernel_initializer='normal', activation='relu')(x)
#                 if dropoutFrac > 0:
#                     x = Dropout(dropoutFrac)(x)
#             
#             # Add output layer.
#             x = Dense(noutputs, kernel_initializer='normal', activation='relu')(x)
#             model = Model(inputTensor, x)
            
            # Sequential API
            model = Sequential()
            model.add(Dense(layerSizes[0], input_shape=(ninputs,),
                            kernel_initializer='normal', activation='relu'))
            # Add additional layers if requested.
            for i in range(nlayers - 1):
                model.add(Dense(layerSizes[i+1], kernel_initializer='normal', activation='relu'))
                if dropoutFrac > 0:
                    model.add(Dropout(dropoutFrac))
            # Add output layer.
            model.add(Dense(noutputs, kernel_initializer='normal', activation='sigmoid'))
            
            model.compile(**compileKwargs)
            return model
        self.makeModel = makeModel
        super(KerasRegressor, self).__init__()
        
    def displayModel(self, model=None):
        if model is None:
            model = self.makeModel()
        from keras.utils.vis_utils import plot_model
        from IPython.display import Image
        p = plot_model(model, to_file='model.png', show_shapes=True)
        return Image(filename='model.png')
    
    @timeMethodDecorator
    def fit(self, X, Y, **kwargs):
        for key, val in dict(nb_epoch=10, batch_size=5, verbose=0).items():
            if key not in kwargs:
                kwargs[key] = val
        self.estimator = keras.wrappers.scikit_learn.KerasClassifier(build_fn=self.makeModel, **kwargs)
        self.estimator.fit(X, Y)

    @timeMethodDecorator
    def predict(self, X=None):
        if X is None: X = self.test.X
        return self.estimator.predict(X, verbose=False).ravel().astype(int)
    
    
class MultiTaskSVR(object):
    
    def __init__(self, ntask, notebook=True, *args, **kwargs):
        for k, v in dict(
            kernel='rbf',
            cache_size=512,
            verbose=False,
            C=1.0,
        ).items():
            if k not in kwargs:
                kwargs[k] = v
        
        self.innerEstimators = [
            svm.SVR(*args, **kwargs)
            for _ in range(ntask)
        ]
        self.ntask = ntask
        self.notebook = notebook
        
    @property
    def progressBar(self):
        if self.notebook:
            return tqdm.tqdm_notebook
        else:
            return tqdm.tqdm
        
    def fit(self, X, Y, progress=True):
        nsamp, _nfeat = X.shape
        nsamp2, ntask2 = Y.shape
        assert nsamp == nsamp2
        assert ntask2 == self.ntask
        
        if progress:
            enumerated = self.progressBar([
                (i, est)
                for (i, est) in enumerate(self.innerEstimators)
                ],
                'fitting %d SVMs' % self.ntask
            )
        else:
            enumerated = enumerate(self.innerEstimators)
        
        for i, est in enumerated:
            est.fit(X, Y[:, i])
        return self
            
    def predict(self, X):
        nsamp, _nfeat = X.shape
        outputs = [
            est.predict(X)
            for est in self.innerEstimators
        ]
        nsamp2 = outputs[0].shape[0]
        assert nsamp == nsamp2
        return np.hstack([
            output.reshape((nsamp, 1))
            for output in outputs
        ])
  
    
class Estimator(object):
    
    def __init__(self, method, *args, **kwargs):
        methods = {
            'svm': MultiTaskSVR,
            'elasticNet': MultiTaskElasticNet,
            'ridge': Ridge,
            'lasso': Lasso,
        }
        self.method = method
        self.constructArgs = args
        self.constructKwargs = kwargs
        self.innerEstimator = methods[method](*args, **kwargs)
        
    def __str__(self):
        out = '%s' % self.method
        if len(self.constructArgs) > 0 or len(self.constructKwargs) > 0:
            args = list(self.constructArgs)
            if len(self.constructKwargs) > 0:
                args.extend(['%s=%s' % (k, v) for (k, v) in self.constructKwargs.items()])
            out += '(%s)' % (', '.join(['%s' % a for a in args]))
        return out
        
    def fit(self, X, Y):
        self.innerEstimator.fit(X, Y)
        return self
    
    def predict(self, X):
        return self.innerEstimator.predict(X)

