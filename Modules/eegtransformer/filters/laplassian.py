import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from copy import deepcopy
from scipy.signal import butter,sosfilt
import matplotlib.pyplot as plt


class Laplassian(BaseEstimator,TransformerMixin):
    def __init__(self,channels_groups,type='small',axis='auto'):
        self.axis=axis
        self.type=type
        self.channels_groups=channels_groups


    def fit(self,X,y=None):
        return self

    def transform(self, X, y = None):

        if self.axis=='auto':
            if np.shape(X)[1]>np.shape(X)[2]:
                X_new=self.__first_axis_avereging(X)
            else:
                X_new=self.__second_axis_avereging(X)
        elif self.axis==1:
            X_new = self.__first_axis_avereging(X)
        elif self.axis==2:
            X_new = self.__second_axis_avereging(X)
        return X_new

    def __first_axis_avereging(self,X):
        transform_matrix=np.eye(np.shape(X)[2],dtype=float)-np.ones(shape=(np.shape(X)[2]),dtype=float)/np.shape(X)[2]
        print(transform_matrix)
        for i in range(np.shape(X)[0]):
            X[i]=np.dot(X[i],transform_matrix)
        # for i in range(np.shape(X)[0]):
        #     average=np.zeros(shape=(1,np.shape(X[i])[0]))
        #     for j in range(np.shape(X)[2]):
        #         average+= np.transpose(X[i,:,j])
        #     average/=np.shape(X)[2]
        #     average=np.transpose(average)
        #     X[i]=np.subtract(X[i],average)
        return X

    def __second_axis_avereging(self,X):
        transform_matrix = np.eye(np.shape(X)[1], dtype=float) - np.ones(shape=(np.shape(X)[1]), dtype=float) / \
                           np.shape(X)[1]
        print(transform_matrix)
        for i in range(np.shape(X)[0]):
            X[i]=np.dot(transform_matrix,X[i])
        # for i in range(np.shape(X)[0]):
        #     average=np.zeros(shape=(1,np.shape(X[i])[1]))
        #     for j in range(np.shape(X)[1]):
        #         average+= X[i,j]
        #     average/=np.shape(X)[1]
        #     X[i]=np.subtract(X[i],average)
        return X

if __name__=="__main__":
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
    print(X_test1.shape)
    cr=CAR()
    res=cr.transform(X_test1)
    print(res)