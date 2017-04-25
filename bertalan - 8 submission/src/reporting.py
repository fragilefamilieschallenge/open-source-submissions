'''
Created on Apr 3, 2017

@author: tsbertalan
'''
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from utilities import _cachedResult

# class Report(object):
#     
#     def evaluate(self, estimator, data):
#         metrics = Metrics(estimator, data)
        
        
class Metrics(object):
    
    def __init__(self, estimator, data, multioutput='uniform_average'):
        self.data = data
        self.estimator = estimator
        self.multioutput = multioutput

    @property
    @_cachedResult 
    def predictionTrain(self):
        Xtrain, _Ytrain = self.data.train
        return self.estimator.predict(Xtrain)

    @property
    @_cachedResult
    def predictionTest(self):
        Xtest, _Ytest = self.data.test
        return self.estimator.predict(Xtest)

    @property
    @_cachedResult
    def r2(self):
        _Xtrain, Ytrain = self.data.train
        _Xtest, Ytest = self.data.test    
        r2_train = r2_score(Ytrain, self.predictionTrain, multioutput=self.multioutput)
        r2_test = r2_score(Ytest, self.predictionTest, multioutput=self.multioutput)
        return {'train': r2_train, 'test': r2_test}

    def responseHistograms(self, responses='Ynum', keys=None, figaxes=None, **kwargs):
        for k, v in {
                'normed': True,
                'alpha': 0.75,
            }.items():
            if k not in kwargs:
                kwargs[k] = v
        if responses is None or responses in (
            'Ynum', 'YnumVal', 'predictionTrain', 'predictionTest'
            ):
            assert keys is None
            keys = self.data.responseKeysTrainedTested
            responses = {
                'Ynum': self.data.Ynum,
                'YnumVal': self.data.YnumVal,
                'predictionTrain': self.predictionTrain,
                'predictionTest': self.predictionTest,
                None: self.data.Ynum,
            }[responses]
        n = responses.shape[1]
        assert n == len(keys)
        if figaxes is None:
            fig = plt.figure(figsize=(9, 4.5))
            axes = [fig.add_subplot(3, n / 3, i+1)
                    for i in range(n)]
        else:
            fig, axes = figaxes
        for i in range(n):
            ax = axes[i]
            ax.hist(responses[:, i], 32, **kwargs)
            ax.set_xlabel(keys[i])
            ax.grid(False)
        fig.tight_layout()
        return fig, axes
    
    def compareAllResponseHistograms(self, **kwargs):
        figaxes = \
        self.responseHistograms(responses='Ynum', label='training data', **kwargs)
        self.responseHistograms(responses='predictionTrain', figaxes=figaxes,
                                label='training predictions', **kwargs);
        self.responseHistograms(responses='YnumVal', figaxes=figaxes,
                                label='testing data', **kwargs)
        self.responseHistograms(responses='predictionTest', figaxes=figaxes,
                                label='testing predictions', **kwargs);
        figaxes[1][1].legend(loc='best')
        return figaxes

        
class MetricsTable(object):
    
    def __init__(self, metrics):
        
        self.metrics = metrics
        
    def __str__(self):
        out = []
        # Begin environments.
        out.append(r'\begin{center}')
        out.append(r'\begin{tabular}{rcc}')
        
        # Headings
        hline = lambda: out.append(r'\hline')
        hline()
        out.append(r' & \textbf{$\hat{r^2}$ (train)} & \textbf{$\hat{r^2}$ (test)} \\')
        hline()
        
        # Content
        for metric in self.metrics:
            out.append(r'\textbf{%s} & %f & %f' % (
                str(metric.estimator).replace('Estimator: ', '')
                .replace('alpha', r'$\alpha$'),
                metric.r2['train'], metric.r2['test']
                )
            )
            out[-1] += r' \\'
        hline()
        
        # End environments
        out.append(r'\end{tabular}')
        out.append(r'\end{center}')
        
        return '\n'.join(out)
    
    def write(self, directory):
        import os.path
        f = file(os.path.join(directory, 'comparisonTable.tex'), 'w')
        f.write(str(self))
        f.close()
                 