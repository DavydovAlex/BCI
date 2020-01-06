import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from copy import deepcopy



class AveragePower(BaseEstimator,TransformerMixin):
    def __init__(self,axis='auto',log=True):
        self.axis=axis
        self.log=log



    def fit(self,X,y=None):
        return self

    def transform(self, X, y = None):

        if self.axis == 'auto':
            if np.shape(X)[1] > np.shape(X)[2]:
                X = (X ** 2).mean(axis=1)
            else:
                X = (X ** 2).mean(axis=2)
        elif self.axis == 1:
            X = (X ** 2).mean(axis=1)
        elif self.axis == 2:
            X = (X ** 2).mean(axis=2)
        if self.log==True:
            X=np.log(X)
        return X



if __name__=='__main__':
    X_test1 = np.array([[[1, 2, 3, 4],
                         [11, 21, 31, 41]],
                       [[4, 5, 6, 7],
                        [41, 51, 61, 1]],
                       [[7, 8, 9, 10],
                        [71, 18, 19, 10]]],dtype=float)
    X_test = np.array([[[1, 2, 3, 4],
                        [11, 21, 31, 41],
                        [11, 21, 31, 41],
                        [11, 21, 31, 41],
                        [41, 51, 61, 1]],
                       [[4, 5, 6, 7],
                        [41, 51, 61, 1],
                        [41, 51, 61, 1],
                        [41, 51, 61, 1],
                        [41, 51, 61, 1]],
                       [[7, 8, 9, 10],
                        [71, 18, 19, 10],
                        [41, 51, 61, 1],
                        [41, 51, 61, 1],
                        [41, 51, 61, 1]]],dtype=float)
    print(X_test.shape)
    cr=AveragePower()
    res=cr.transform(X_test)
    print(res)