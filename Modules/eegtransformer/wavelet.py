import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from copy import deepcopy
from scipy.signal import butter,sosfilt
import matplotlib.pyplot as plt
from scipy import signal

class Wavelet(BaseEstimator,TransformerMixin):
    def __init__(self,widths=np.arange(1,31)):
        self.widths=widths

    def fit(self,X,y=None):
        return self

    def transform(self, X, y = None):

        bufData=np.zeros(shape=(X.shape[0],X.shape[1]*len(self.widths)*X.shape[2]),dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                bufData[i,j*X.shape[1]*len(self.widths):j*X.shape[1]*len(self.widths)+X.shape[1]*len(self.widths)]=np.ravel(signal.cwt(X[j,:,j], signal.ricker, self.widths))





        return bufData


if __name__=="__main__":
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from eegreader import EEGData
    from sklearn.pipeline import Pipeline
    from sklearn import svm
    from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold

    EEG=EEGData()
    wv=Wavelet()
    # wv.transform(EEG.EEGData
    svm = svm.SVC(kernel='rbf', C=1)
    union=Pipeline([('wv',wv),('svm',svm)])
    scores = cross_val_score(union, EEG.EEGData, EEG.Labels, cv=5)
    print(scores)