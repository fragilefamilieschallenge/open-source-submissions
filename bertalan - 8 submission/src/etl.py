'''
Extract, Transform, Load

Created on Apr 2, 2017

@author: tsbertalan
'''


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, tqdm
from utilities import timeit, getDtype, isCategory

import os
os.environ["PATH"] += os.pathsep + '/usr/local/cuda-8.0/bin/'
import fancyimpute

def loadRawData(csvPath):
    pass


class Data(object):
    
    def __init__(self,
                 dataDir='../data/',
                 notebook=False,
                 nanThresh=0.5,
                 loadIfPossible='preprocessed.npz',
                 debug=False,
                 imputationMethod='most_frequent',
                 imputationMethodY='MICE',
                 nval=121,
        ):
        '''Container object for training, test/validation, and challenge data.
        
        Parameters
        ==========
            dataDir : str, optional
                Path to folder holding .csv and .npz files.
            notebook : bool, optional
                Whether to use Jupyter-notebook friendly progress bars.
            nanThresh : float, optional
                Fraction of values in a column which may be NaN, above which
                the column is dropped.
            loadIfPossible : bool or string, optional
                Whether to load preprocessed data from .npz file instead
                of preprocessing and saving it (and saving to npz file) here and now,
                and, if so, what file in dataDir to load.
            debug : bool, optional
                When preprocessing from scratch, only load a small subset of the rows
                in the full dataset.
            nval : int, optional
                How many training samples to dedicate to validation.
        
        ''' 
        self.nanThresh = nanThresh
        self.dataDir = dataDir
        self.notebook = notebook
        self.debug = debug
        self.nval = nval
        self.imputationMethod = imputationMethod
        self.imputationMethodY = imputationMethodY
        
        if notebook:
            self.tqdm = tqdm_notebook
        else:
            self.tqdm = tqdm
        
        ## Consider loading from a savefile instead of preprocessing from scratch.
        if loadIfPossible:
            if isinstance(loadIfPossible, str):
                fname = loadIfPossible
            else:
                fname = 'preprocessed.npz'
            if fname == 'preprocessed.npz' and debug:
                fname = 'preprocesssed_debug.npz'
            fullPath = os.path.join(dataDir, fname)
            if os.path.isfile(fullPath):
                # We found a savefile and we're directed to use it.
                with timeit('loading data'):
                    print 'Loading from %s ...' % fullPath,
                    loaded = np.load(fullPath, allow_pickle=True)
                    # Unpack single-object 0-dimensional object arrays.
                    for k in loaded.keys():
                        lk = loaded[k]
                        if (
                            lk.size == 1
                            and len(lk.shape) == 0
                            ):
                            lk = lk.reshape((1,))[0]
                        elif (
                            lk.dtype == object
                            and len(lk.shape) == 1
                            ):
                            lk = lk.tolist()
                        self.__dict__[k] = lk
                    print ' done.'
                    
                # If we got here, we don't need to do the remaining constructor steps.
                return None
        
        ## Load CSV files.
        # Features.
        self.features = pd.read_csv(
            os.path.join(dataDir, 'background.csv'),
            low_memory=False
        )
        
        # Training labels.
        self.responses = pd.read_csv(
            os.path.join(dataDir, 'train.csv')
        )
        
            
        if debug:
            self.features = self.features[:256]
            self.responses = self.responses[:256]
            

        ## Preprocess.
        self._preprocess()
        
    def _preprocess(self):
        with timeit('preprocessing'):
            self._writeParametersFile()
            self._removeBadColumns()
            self._applyCodebooks()
            self._recodeNaNs()
            self._recodeCategoricals()
            self._discardUndocumented()
            self._oneHot()
            self._matchResponses()
            self._recodeResponses()
            self._makeArrays()
            self._impute()
            self._standardize()
            self._savemats()
    
    def _writeParametersFile(self):
        f = open(os.path.join(self.dataDir, '..', 'doc', 'parameterValues.tex'), 'w')
        for name, value in (
            ('nval', self.nval),
            ('nanThresh', self.nanThresh),
            ('imputationMethod', self.imputationMethod),
            ('imputationMethodY', self.imputationMethodY),            
                            ):
            if name == 'imputationMethod':
                value = {
                    'most_frequent': 'mode',
                    'mean': 'mean',
                    'median': 'median'
                }[value]
            f.write(r'\newcommand{\%s}{%s}' % (name, value) + '\n')
        f.close()
    
    def _removeBadColumns(self):
        # Remove mostly-NaN and nonvarying columns .
        keys =  [str(k) for k in self.features.keys()]
        keys.sort()
        nanFracs = []
        for k in self.tqdm(keys, desc='Tabulate NaN fractions'):
            nanFracs.append(
                float(sum(pd.isnull(
                    self.features[k]
                ))) / self.features[k].size
            )
        
        fig, ax = plt.subplots()
        a = .5
        ax.hist(nanFracs, bins=32, alpha=a, label='all fractions', normed=True)
        ax.hist([f for f in nanFracs if f != 0], alpha=a, label='nonzero fractions', normed=True)
        ax.legend(loc='best')
        ax.set_xlabel('fraction of null values')
        ax.set_ylabel('normalized histogram')
        fig.savefig(os.path.join(self.dataDir, '..', 'doc', 'nanHistogram.pdf'))
        
        badKeys = set([
            k for (k, f) in zip(keys, nanFracs)
            if f > self.nanThresh
        ])
        print '%d of %d columns have >%.1f%% NaN values' % (
            len(badKeys), len(keys), self.nanThresh*100.
        )
        
        # Additionally, leave out constant columns.
        keys = [str(k) for k in self.features.keys()]
        keys.sort()
        nvalsPerKey = []
        for k in keys:
            nvalsPerKey.append(len(self.features[k].unique()))
            
        nbad = (np.array(nvalsPerKey) == 1).sum()
        ntot = len(keys)
        
        print '%d of %d keys columns are constant.' % (nbad, ntot)
        
        for n, k in zip(nvalsPerKey, keys):
            if n == 1:
                badKeys.add(k)
                
        before = len(self.features.keys()) 
        self.features.drop(badKeys, axis=1, inplace=True)
        after = len(self.features.keys())
        print '%d keys dropped (%.1f%%).' % (before - after, float(before - after) / before * 100.)
                
    def _applyCodebooks(self):
        self.allVars = {}
        for f in '''ff_child_cb9.txt  ff_dad_cb5.txt  ff_mom_cb3.txt      ff_teacher_cb9.txt
        ff_dad_cb0.txt    ff_dad_cb9.txt  ff_mom_cb5.txt
        ff_dad_cb1.txt    ff_mom_cb0.txt  ff_mom_cb9.txt
        ff_dad_cb3.txt    ff_mom_cb1.txt  ff_teacher_cb5.txt'''.split():
            f = f.strip()
            self.allVars.update(parseCodebook('../data/codebooks/%s' % f))
            
        self.keys = [k for k in self.allVars.keys() if k in self.features]

    def getDescription(self, key):
        return self.allVars[key]['description']
    
    def _recodeNaNs(self):
        '''Replace NaNs or Other with -3 ("Missing").'''
        self.features.replace({
            'Other': -3,
            '<0.1': 0.999999,
        }, regex=False, inplace=True)
        
        self.features.fillna(value=-3, inplace=True)
        
    def _recodeCategoricals(self):
        '''Convert byte variables to categorical.'''
        self.types = [self.allVars[k]['type'] for k in self.keys]
        enumeration = [(i,k) for (i,k) in enumerate(self.keys) if self.types[i] is str]
        for i, k in self.tqdm(enumeration, desc='Recode categoricals'):
            if self.types[i] is str:
                self.features[k] = self.features[k].astype('category')
                
    def _discardUndocumented(self):
        '''Only keep the features documented in the codebooks.'''
        def append(l, x):
            out = list(l)
            out.append(x)
            return out
        self.features = pd.DataFrame({
                k: self.features[k]
                for k in append(self.keys, 'challengeID')
            })
        
    def _oneHot(self):
        self.features = pd.get_dummies(
            self.features,
            columns=[k for (i,k) in enumerate(self.keys) if self.types[i] is str],
            drop_first=True,  # Use k-1 one-hot columns to eoncode k categories.
        )
        
    def _matchResponses(self):
        '''Separate training and challenge features.
        
        Training features are those with corresponding response values.'''
        self.merged = pd.merge(self.features, self.responses, on='challengeID')
        
        # The unmerged and merged are disjoint.
        self.unmerged = self.features[np.logical_not(self.features.challengeID.isin(self.merged.challengeID))]
        overlap = pd.merge(self.merged, self.unmerged, on='challengeID')
        assert len(overlap) == 0
        
        # And the counts make sense.
        if not self.debug:
            assert len(self.merged) == len(self.responses)
            assert len(self.merged) + len(self.unmerged) == len(self.features)
        
        # The keys which are not in the unmerged data are the resposne variables.
        self.mkeys = set(self.merged.keys())
        self.ukeys = set(self.unmerged.keys())
        self.responseKeys = list(self.mkeys - self.mkeys.intersection(self.ukeys))
        self.responseKeys.sort()
        self.mkeys = list(self.mkeys); self.mkeys.sort()
        self.ukeys = list(self.ukeys); self.ukeys.sort()
        
        # At this point, we're done with features, so we'll delete it to regain about 6 GB.
        del self.features, self.responses
        
    def _recodeResponses(self):
        '''Process response variables also (NaNs and categoricals).'''
        # I don't know of a simpler way to extract the category type.
        # Though it's undoubtedly available in the package directly somewhere.
        self.merged.fillna(value=-3, inplace=True)
        for k in self.responseKeys:
            if self.merged[k].dtype is np.dtype(object) or self.merged[k].dtype is getDtype('bool'):
                self.merged[k] = self.merged[k].astype('category')
                
    def _makeArrays(self):
        '''Replace Pandas representation with Numpy arrays.'''
        Xtrain = self.merged.as_matrix(columns=self.ukeys).astype(float)
        print 'Train predictors are shape', Xtrain.shape
        
        Xchallenge = self.unmerged.as_matrix(columns=self.ukeys).astype(float)
        print 'Challenge predictors shape', Xchallenge.shape
 
        self.responseTypes = [
            self.merged[k].dtype
            for k in self.responseKeys
        ]
        
        Ytrain = self.merged.as_matrix(columns=self.responseKeys).astype(float)
        print 'Train responses are shape', Ytrain.shape
        
        self.idTrain = self.merged['challengeID']
        self.idTest  = self.unmerged['challengeID']
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xchallenge = Xchallenge
        
        # At this point, we're done with merged and unmerged, so we'll delete them to regain about 3.3 GB each.
        del self.merged
        del self.unmerged
    
    def _impute(self, n_nearest_columns=12):
        xmethod = self.imputationMethod
        ymethod = self.imputationMethodY
        def Imputer(method):
            if method == 'MICE':
                imputer = fancyimpute.MICE(
                    n_nearest_columns=n_nearest_columns,
                    min_value=0.0,
                    verbose=False,
                )
                def impute(data):
                    return imputer.complete(data)
            elif method in ('fancymean', 'zero', 'fancymedian', 'min', 'random'):
                imputer = fancyimpute.SimpleFill(
                    min_value=0.0,
                    fill_method=method,
                )
                def impute(data):
                    return imputer.complete(data)
            elif method in ('mean', 'median', 'most_frequent'):
                import sklearn.preprocessing
                imputer = sklearn.preprocessing.Imputer(
                    strategy=method,
                )
                def impute(data):
                    return imputer.fit_transform(data)
            elif method == 'drop':
                def impute(data):
                    raise NotImplementedError
                    
            return impute
            
        XtrainShape = self.Xtrain.shape
        #XchallengeShape = self.Xchallenge.shape
        Xfull = np.vstack((self.Xtrain, self.Xchallenge))
        del self.Xtrain, self.Xchallenge
        import gc
        gc.collect()
            
        with timeit('imputing predictors by %s' % xmethod):
            imputer = Imputer(xmethod)
            Xfull[Xfull < 0] = np.nan
            print '%d NaNs.' % np.isnan(Xfull).sum()
            Xfull = imputer(Xfull)
            print '%d NaNs.' % np.isnan(Xfull).sum()
        
        self.Xtrain = Xfull[:XtrainShape[0], :]
        self.Xchallenge = Xfull[XtrainShape[0]:, :]
        
        
        self.Ytrain[self.Ytrain < 0] = np.nan
        if ymethod == 'drop':
            # Drop rows with NaN in response variables.
            keepRows = np.logical_not(np.isnan(self.Ytrain).sum(1).astype(bool))
            print 'Keeping %d training rows out of %d with no missing response values.' % (keepRows.sum(), len(keepRows)) 
            self.Ytrain = self.Ytrain[keepRows, :]
            self.Xtrain = self.Xtrain[keepRows, :]
            self.idTrain = self.idTrain[keepRows]
        else:
            with timeit('imputing Ytrain by %s' % ymethod):
                imputer = Imputer(ymethod)
                print '%d NaNs.' % np.isnan(self.Ytrain).sum()
                self.Ytrain = imputer(self.Ytrain)
                print '%d NaNs.' % np.isnan(self.Ytrain).sum() 
        
    def _standardize(self):
        '''Use mean-variance standardization.'''
        
    def _savemats(self, fname='preprocessed.npz'):
        if fname == 'preprocessed.npz' and self.debug:
            fname = 'preprocessed_debug.npz'
        fullPath = os.path.join(self.dataDir, fname)
        print 'Saving to %s.' % fullPath
        np.savez_compressed(
            fullPath,
            **{
                k: v
                for (k, v) in self.__dict__.items()
                if k not in ['tqdm',]
                and k[0] != '_'
                }
        )
 
    @property
    def XtrainSubset(self):
        return self.Xtrain[:-self.nval, :]
 
    @property
    def YtrainSubset(self):
        return self.Ytrain[:-self.nval, :]
 
    @property
    def XvalSubset(self):
        return self.Xtrain[-self.nval:, :]
 
    @property
    def YvalSubset(self):
        return self.Ytrain[-self.nval:, :]
    
    @property
    def notCatResp(self):
        return [not isCategory(t) for t in self.responseTypes]

    @property
    def Ynum(self):
        return self.YtrainSubset[:, self.notCatResp].astype(float)
    
    @property
    def YnumVal(self):
        return self.YvalSubset[:, self.notCatResp].astype(float)
    
    @property
    def train(self):
        return self.XtrainSubset, self.Ynum
    
    @property
    def test(self):
        return self.XvalSubset, self.YnumVal
    
    @property
    def responseKeysTrainedTested(self):
        return [
            k
            for (i, k) in enumerate(self.responseKeys)
            if self.notCatResp[i]
        ]
        
        
def parseCodebook(fpath):
    codebook = open(fpath).read()
    headingMode = False
    data = {}
    dataItem = {}
    for l in codebook.split('\n')[1:]:
        # Ignore lines which are empty or start with a period.
        if len(l.strip()) == 0 or l[0] == '.':
            continue

        # Switch between two major modes on '----'-ish lines: heading and data.
        if '-'*10 in l:  # We saw a divider line.
            headingMode = not headingMode
            continue

        # Parse headings.
        if headingMode:
            # Save previous data.
            if len(dataItem) > 0:
                data[key] = dataItem
                data[key]['description'] = desc

            # Now, reset state for the next dataItem.
            dataItem = {}
            key = l.split()[0]
            desc = ' '.join(l.split()[1:])

            continue

        # Skip some lines.
        if (
            ('missing' in l and 'unique' in l)
            or 'opened on' in l
            or 'log type' in l
            ):
            
            continue

        # In fact, only accept a small subset of lines.
        if 'type' not in l:
            continue

        # Parse data.
        if not headingMode:
            split = l.split(':')
            split2 = split
            assert len(split) in (1, 2, 3), split
            if len(split) == 3:
                split23 = split[1].split()
                split2 = [split[0]]
                split2.extend(split23)
                split2.append(split[-1])

            split2 = [x.strip() for x in split2]

            for i in range(0, 10, 2):
                if len(split2) >= i+2:
                    dataKey = split2[i].strip()
                    dataVal = split2[i+1].strip()
                    baseTypes = {'int': int, 'float': float, 'double': float, 'str': str, 'byte': str, 'long': int}
                    for k, v in baseTypes.items():
                        if k in dataVal:
                            dataVal = v
                            break
                    dataItem[dataKey] = dataVal


    # Save previous data.
    if len(dataItem) > 0:
        data[key] = dataItem
        data[key]['description'] = desc

    return data


if __name__ == '__main__':
    
    data = Data(debug=True, loadIfPossible=False)
