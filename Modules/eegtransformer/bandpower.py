import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from copy import deepcopy



class BandPower(BaseEstimator,TransformerMixin):
    def __init__(self,band,frequency, value='bandPower'):
        self.band=band
        self.frequency=frequency
        self.value=value



    def fit(self,X,y=None):
        return self

    def transform(self, X, y = None):

        bp=np.empty(shape=(np.shape(X)[0],np.shape(X)[2]))
        max = np.empty(shape=(np.shape(X)[0], np.shape(X)[2]))
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[2]):
                bp[i,j],max[i,j]=self.bandpower(X[i,:,j],self.frequency,self.band)
        if self.value=='bandPower':
            result=bp
        elif self.value=='maxPower':
            result=max
        elif self.value=='all':
            result=np.concatenate((bp,max))

        return result


    def bandpower(self,data, sf, band, window_sec=None, relative=False):
        """Compute the average power of the signal x in a specific frequency band.

        Parameters
        ----------
        data : 1d-array
            Input signal in the time-domain.
        sf : float
            Sampling frequency of the data.
        band : list
            Lower and upper frequencies of the band of interest.
        window_sec : float
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative : boolean
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.

        Return
        ------
        bp : float
            Absolute or relative band power.
        """
        from scipy.signal import welch
        from scipy.integrate import simps
        # band = np.asarray(band)
        low=band[0]
        high=band[1]

        nperseg = np.shape(data)[0]
        # Define window length
        # if window_sec is not None:
        #     nperseg = window_sec * sf
        # else:
        #     nperseg = (2 / low) * sf

        # Compute the modified periodogram (Welch)
        freqs, psd = welch(data, sf, nperseg=nperseg)

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]

        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Integral approximation of the spectrum using Simpson's rule.
        bp = simps(psd[idx_band], dx=freq_res)
        max=np.ndarray.max(psd[idx_band])
        if relative:
            bp /= simps(psd, dx=freq_res)
        return bp,max

if __name__=='__main__':
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from eegreader import EEGData

    bp=BandPower([8,12],frequency=250,value='maxPower')
    EEG=EEGData()
    bp=bp.transform(EEG.EEGData)
    print(bp)