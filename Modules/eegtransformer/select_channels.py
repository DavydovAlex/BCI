import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

"""

Выбор каналов ЭЭГ

"""
class SelectChannels(BaseEstimator, TransformerMixin):

    def __init__(self, ordered_channels=(), selected_channel=(),montage_type='encefalan_19', axis='auto'):
        """
        Args:
            ordered_channels (tuple): Последовательность электодов в монтаже по умолчанию
            selected_channel (tuple): Электроды для выбора
            montage_type     (str)  : Тип монтажа. Добавляется в функции 'montage()'
            axis             (int or str)  : Ось, содержащая каналы ЭЭГ
        """
        self.selected_channel = selected_channel
        self.axis = axis
        if ordered_channels:
            self.ordered_channels = ordered_channels
        else:
            montage = self.montage(montage_type)
            if montage:
                self.ordered_channels = montage
            else:
                raise ValueError("'ordered_channels' is empty and 'montage_type' not valid")

    def fit(self,X,y=None):
        return self

    def transform(self, X, y = None):
        order=[] #Упорядочивание элементов в кортеже выбираемых значений
        for default_val in self.ordered_channels:
            for selected_val in self.selected_channel:
                if default_val==selected_val:
                    order.append(default_val)
        print(order)
        indexes = [self.ordered_channels.index(channel) for channel in order]

        # Поиск наименьшего измерения
        #
        # В нем содержится число каналов
        if self.axis=='auto':
            if np.shape(X)[1]>np.shape(X)[2]:
                X=X[:, :, indexes]
            else:
                X = X[:, indexes, :]
        elif self.axis==1:
            X = X[:, indexes, :]
        elif self.axis==2:
            X = X[:, :, indexes]

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