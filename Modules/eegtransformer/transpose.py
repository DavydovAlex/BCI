import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin


class Transpose(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    def transform(self, X, y = None):
        print(np.shape(X))
        X_transposed=np.zeros(shape=(np.shape(X)[0],np.shape(X)[2],np.shape(X)[1]), dtype=float)

        for i in range(X.shape[0]):
            X_transposed[i,:,:]=np.matrix.transpose(X[i,:,:])

        return X_transposed


if __name__=="__main__":
    X_test = np.array([[[1, 2, 3], [1, 2, 3]],
                       [[4, 5, 6], [4, 5, 6]],
                       [[7, 8, 9], [7, 8, 9]]])
    tr=Transpose()
    res=tr.transform(X_test)
    print(res) 
