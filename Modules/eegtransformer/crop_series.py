import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from copy import deepcopy


class CropSeries(BaseEstimator,TransformerMixin):
    def __init__(self, frequency, start_time, seria_length, axis='auto'):
        """
        Args:
            frequency (int): Частота дискретизации
            start_time(int): Начало обрезки сигнала в секундах
            seria_length(int): Длина результирующего сигнала в секндах
            axis(int,str): Ось по которой производится обрезка. По умолчанию большая
        """
        self.frequency=frequency
        self.start_time=start_time
        self.seria_length=seria_length
        self.axis=axis


    def fit(self,X,y=None):
        return self

    def transform(self, X, y = None):
        start = self.start_time
        frq = self.frequency
        lenght = self.seria_length

        if self.axis=='auto':
            if np.shape(X)[1]>np.shape(X)[2]:
                X=X[:, start*frq : start*frq+lenght*frq, :]
            else:
                X = X[:,:, start * frq : start * frq + lenght * frq]
        elif self.axis==1:
            X = X[:, start*frq : start*frq+lenght*frq, :]
        elif self.axis==2:
            X = X[:,:, start * frq : start * frq + lenght * frq]
        return X


if __name__=="__main__":
    X_test=np.array([[[1,2],
                      [11,21],
                      [12,22],
                      [13,23],
                      [14,24],
                      [15,25]],
                     [[4,5],
                      [41,51],
                      [42,52],
                      [43,53],
                      [44,54],
                      [45,55]]])
    cropper=CropSeries(1,2,1)
    print(cropper.transform(X_test))

