'''
Created on Apr 3, 2017

@author: tsbertalan
'''
import numpy as np

class SpectralSelector(object):
    
    def __init__(self, data, ncomps=128, nystrommed=False, distanceMethod=True, pca=False):
        
        self.data = data
        self.nystrommed = nystrommed
        
        # Nystroemmable codes
        if nystrommed:
            import sklearn.kernel_approximation
            if True:
                dimRes = sklearn.kernel_approximation.Nystroem(
                    n_components=ncomps, random_state=4,
                    kernel='rbf'
                )
            else:
                dimRes = sklearn.kernel_approximation.RBFSampler(
                    n_components=ncomps, random_state=4,
                )
            # Doesn't work (doesn't allow negative inputs?)
        #     dimRes = sklearn.kernel_approximation.SkewedChi2Sampler(
        #         n_components=ncomps, random_state=4,
        #     )
        
        # Non-Nystroem'd
        else:
            import sklearn.manifold
            dimRes = sklearn.manifold.SpectralEmbedding(
                n_components=ncomps, random_state=4,
                affinity='rbf', n_jobs=-1
            )
            
        self.dimRes = dimRes
    
    def fitTransformNoExtension(self, dimRes, Xall=None):
        if Xall is None:
            Xall = self.data.Xtrain
        transformed = dimRes.fit_transform(Xall)
        return transformed[:-self.data.nval, :], transformed[-self.data.nval:, :]
    
    def fit_transform(self, Xall=None, Y=None):
        if self.nystrommed:
            if Xall is None:
                Xa = self.data.XtrainSubset
                Xb = self.data.Xvalsubset
            else:
                Xa = Xall[:-self.data.nval, :]
                Xb = Xall[-self.data.nval:, :]
            XtrainTransformed = self.dimRes.fit_transform(Xa)
            XvalTransformed = self.dimRes.transform(Xb)
        else:
            XtrainTransformed, XvalTransformed = self.fitTransformNoExtension(self.dimRes, Xall)
        
        return np.vstack([XtrainTransformed, XvalTransformed])