import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

"""

Выбор каналов ЭЭГ

"""
class SelectChannels(BaseEstimator, TransformerMixin):

    def __init__(self,labels, selected_classes=(), axis='auto'):

        self.labels=labels
        self.selected_classes = selected_classes
        self.axis = axis


    def fit(self,X,y=None):
        return self

    def transform(self, X, y = None):


        return X

    def montage(self, montage_type):
        if montage_type=='encefalan_19':
            channels_order=("O2", "O1", "P4", "P3", "C4", "C3", "F4", "F3", "Fp2", "Fp1", "T6", "T5", "T4", "T3", "F8",
                            "F7", "Pz", "Cz", "Fz")
            return channels_order
        else:
            return ()

if __name__=="__main__":
    X_test=np.array([[[1,2,3],[1,2,3],[1,2,3]],
                     [[4,5,6],[4,5,6],[4,5,6]],
                     [[7,8,9],[7,8,9],[7,8,9]]])
    defChannels=('A1','A2','A3')
    selChannelsOrdered=('A1','A3')
    selChannelsUnordered=('A3','A2')

    scOrdered=SelectChannels(defChannels,selChannelsOrdered,axis=2)
    print ('Ordered array')
    print (scOrdered.transform(X_test)  )

    scUnordered = SelectChannels(defChannels, selChannelsUnordered)
    print('Unordered array')
    print(scUnordered.transform(X_test))