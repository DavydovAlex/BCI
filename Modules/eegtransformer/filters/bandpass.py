import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from copy import deepcopy
from scipy.signal import butter,sosfilt
import matplotlib.pyplot as plt


class Bandpass(BaseEstimator,TransformerMixin):
    def __init__(self,frequency,lowF,highF,order=5,axis='auto'):
        self.order=order
        self.lowF=lowF
        self.highF = highF
        self.frequency=frequency
        self.axis=axis


    def fit(self,X,y=None):
        return self

    def transform(self, X, y = None):

        # X_copy=deepcopy(X)

        NormFreq = self.frequency * 0.5
        low = self.lowF / NormFreq
        high = self.highF / NormFreq
        sos = butter(self.order, [low, high], btype='band', output='sos')
        if self.axis=='auto':
            if np.shape(X)[1]>np.shape(X)[2]:
                X=sosfilt(sos, X,1)
            else:
                X = sosfilt(sos, X,2)
        elif self.axis==1:
            X = sosfilt(sos, X, 1)
        elif self.axis==2:
            X = sosfilt(sos, X, 2)
        return X





if __name__=="__main__":




    t = np.linspace(0, 1, 1000, False)  # 1 second
    sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)
    print(sig.shape)
    sample=np.array([sig, sig, sig, sig]).T
    print(sample.shape)
    X_test=np.array([sample,sample])
    print(X_test.shape)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, sig)
    ax1.set_title('10 Hz and 20 Hz sinusoids')
    ax1.axis([0, 1, -2, 2])

    flt=Bandpass(1000,15,25,5)
    filtered=flt.transform(X_test)
    print(filtered.shape)
    ax2.plot(t, filtered[0,:,0])
    ax2.set_title('After 15 Hz high-pass filter')
    ax2.axis([0, 1, -2, 2])
    ax2.set_xlabel('Time [seconds]')
    plt.tight_layout()
    plt.show()
