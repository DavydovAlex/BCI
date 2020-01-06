import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion

"""

Снижениче частоты дискретизации по указанному измерению

Attributes:
    new_frequency (int): Частота после децимации
"""
class DownSample(BaseEstimator,TransformerMixin):
    def __init__(self,old_frequency,order=1,axis='auto'):
        """
        Args:
            old_frequency(int): Частота дискретизации
            order(int): Порядок децимации
            axis(int,str): Ось по которой производится децимация
        """
        self.old_frequency=old_frequency
        self.order = order
        self.axis=axis
        self.new_frequency=old_frequency//order

    def fit(self,X,y=None):
        return self

    def transform(self, X, y = None):

        if self.axis=='auto':
            if np.shape(X)[1]>np.shape(X)[2]:
                X_new=self._first_dim(X)
            else:
                X_new=self._second_dim(X)
        elif self.axis==1:
            X_new = self._first_dim(X)
        elif self.axis==2:
            X_new = self._second_dim(X)

        return X_new

    def _first_dim(self,X):
        X_new = np.zeros(
            shape=(np.shape(X)[0], np.shape(X)[1] // self.order, np.shape(X)[2]),
            dtype=float)
        iterator = 0  # счетчик для выбора нужной строки
        counter = 0  # счетчик для определения новой позиции строки
        for i in range(0, (np.shape(X)[1] // self.order) * self.order):
            if iterator == self.order - 1:
                iterator = 0
                X_new[:, counter, :] = X[:, i, :]
                counter += 1
                continue
            iterator += 1
        return X_new

    def _second_dim(self,X):
        X_new = np.zeros(
            shape=(np.shape(X)[0], np.shape(X)[1], np.shape(X)[2]// self.order),
            dtype=float)
        iterator = 0  # счетчик для выбора нужной строки
        counter = 0  # счетчик для определения новой позиции строки
        for i in range(0, (np.shape(X)[2] // self.order) * self.order):
            if iterator == self.order - 1:
                iterator = 0
                X_new[:, :, counter] = X[:, :, i]
                counter += 1
                continue
            iterator += 1
        return X_new


if __name__=="__main__":
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from eegreader import EEGData
    import transpose
    EEG=EEGData()
    ds=DownSample(250,3)
    tr = transpose.Transpose()
    p=Pipeline([('tr',tr),('ds',ds)])

    res=p.transform(EEG.EEGData)
    print(np.shape(res))
    print(ds.new_frequency)