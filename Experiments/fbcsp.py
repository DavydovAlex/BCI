import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..','Modules'))
from eegtransformer import SelectChannels,CropSeries,Bandpass,Transpose,BandPower
from eegreader import EEGData
import numpy as np
from sklearn.pipeline import Pipeline,FeatureUnion
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score,KFold
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator,TransformerMixin


class FBCSP(BaseEstimator,TransformerMixin):
    def __init__(self,n_components=1,filterDiaposones=[[1,40]],frequency=250, transform_into='average_power', log=True ):
        self.n_components=n_components
        self.filterDiaposones=filterDiaposones
        # self.IsModificate=0
        self.frequency=frequency
        # print(transform_into)
        self.transform_into = transform_into
        if transform_into=='csp_space':
            self.tranform_info_CSP='csp_space'
        else:
            self.tranform_info_CSP = 'average_power'
        self.transform_into=transform_into
        if transform_into == 'average_power':
            if log is not None and not isinstance(log, bool):
                raise ValueError('log must be a boolean if transform_into == '
                                 '"average_power".')
        else:
            if log is not None and transform_into=='csp_space':
                raise ValueError('log must be a None if transform_into == '
                                 '"csp_space".')
        # if transform_into=='band_power':
        #     self.transform_into = 'average_power'
        #     self.IsModificate = 1
        # elif transform_into=='max_power':
        #     self.transform_into = 'average_power'
        #     self.IsModificate = 2
        # elif transform_into=='all':
        #     self.transform_into = 'average_power'
        #     self.IsModificate = 3
        # else:
        #     self.transform_into=transform_into



        self.log = log

    def fit(self,X,y):
        filtersCount = np.shape(self.filterDiaposones)[0]

        filters = []
        CSPs = []
        Pipelines = []
        FB = []
        for i in range(filtersCount):
            lowF = self.filterDiaposones[i][0]

            highF = self.filterDiaposones[i][1]

            filters.append(Bandpass(self.frequency, lowF, highF,axis=2))
            CSPs.append(CSP(n_components=self.n_components, transform_into=self.tranform_info_CSP, reg=None, log=self.log))
            Pipelines.append(Pipeline([('filter_' + str(i), filters[i]), ('csp_' + str(i), CSPs[i])]))
            # Pipelines.append(Pipeline([('filter_' + str(i), filters[i])]))
            FB.append(('F_' + str(i), Pipelines[i]))
        FU = FeatureUnion(FB)

        FU.fit(X, y)
        CSPfilters=[]
        for i in range(filtersCount):
            CSPfilters.append(CSPs[i].filters_[:self.n_components])

        self.CSPfilters=CSPfilters
        buf = np.empty(shape=(np.shape(X)[0], self.n_components * filtersCount, np.shape(X)[2]))
        for i in range(filtersCount):
            buf[:, i * self.n_components:i * self.n_components + self.n_components] = np.asarray(
                [np.dot(self.CSPfilters[i], epoch) for epoch in filters[i].transform(X)])

        X =buf

        # compute features (mean band power)
        X = (X ** 2).mean(axis=2)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self



    def transform(self,X):
        filtersCount = np.shape(self.filterDiaposones)[0]
        filters = []
        buf = np.empty(shape=(np.shape(X)[0], self.n_components * filtersCount, np.shape(X)[2]))
        for i in range(filtersCount):
            lowF = self.filterDiaposones[i][0]
            highF = self.filterDiaposones[i][1]
            filter=Bandpass(self.frequency, lowF, highF, axis=2)
            buf[:, i * self.n_components:i * self.n_components + self.n_components] = np.asarray(
                [np.dot(self.CSPfilters[i], epoch) for epoch in filter.transform(X)])
        filtersCount = np.shape(self.filterDiaposones)[0]


        if self.transform_into=='average_power':
            buf = (buf ** 2).mean(axis=2)
            log = True if self.log is None else self.log
            if log:
                buf = np.log(buf)
            else:
                buf -= self.mean_
                buf /= self.std_
        if self.transform_into=='band_power':
            band=np.empty(shape=(np.shape(X)[0],self.n_components * filtersCount*filtersCount))
            for i in range(filtersCount):
                bp=BandPower(self.filterDiaposones[i],self.frequency,'bandPower')
                tr=Transpose()
                pipe=Pipeline([('tr',tr),('bp',bp)])

                band[:,i * self.n_components*filtersCount:i * self.n_components*filtersCount + self.n_components*filtersCount]= \
                    pipe.transform(buf)
            buf = band
        elif self.transform_into=='max_power':
            band = np.empty(shape=(np.shape(X)[0], self.n_components * filtersCount * filtersCount))
            for i in range(filtersCount):
                bp = BandPower(self.filterDiaposones[i], self.frequency, 'maxPower')
                tr = Transpose()
                pipe = Pipeline([('tr', tr), ('bp', bp)])

                band[:,
                i * self.n_components * filtersCount:i * self.n_components * filtersCount + self.n_components * filtersCount] = \
                    pipe.transform(buf)
            buf = band
        elif self.transform_into=='all':
            var = (buf ** 2).mean(axis=2)
            log = True if self.log is None else self.log
            if log:
                var = np.log(var)
            else:
                var -= self.mean_
                var /= self.std_
            print(np.shape(var))
            band = np.empty(shape=(np.shape(X)[0], self.n_components * filtersCount * filtersCount))
            for i in range(filtersCount):
                bp = BandPower(self.filterDiaposones[i], self.frequency, 'bandPower')
                tr = Transpose()
                pipe = Pipeline([('tr', tr), ('bp', bp)])

                band[:,
                i * self.n_components * filtersCount:i * self.n_components * filtersCount + self.n_components * filtersCount] = \
                    pipe.transform(buf)
            print(np.shape(band))
            max = np.empty(shape=(np.shape(X)[0], self.n_components * filtersCount * filtersCount))
            for i in range(filtersCount):
                bp = BandPower(self.filterDiaposones[i], self.frequency, 'maxPower')
                tr = Transpose()
                pipe = Pipeline([('tr', tr), ('bp', bp)])

                max[:,
                i * self.n_components * filtersCount:i * self.n_components * filtersCount + self.n_components * filtersCount] = \
                    pipe.transform(buf)
            print(np.shape(max))
            buf = np.concatenate((var,band,max),axis=1)
            print(np.shape(buf))
        return buf

if __name__=='__main__':
    EEG=EEGData()
    defaultCnls = ["O2", "O1", "P4", "P3", "C4", "C3", "F4", "F3", "Fp2", "Fp1", "T6", "T5", "T4", "T3", "F8", "F7",
                   "Pz", "Cz", "Fz"]

    gs_params={'FBCSP__n_components':range(5,12),
               'FBCSP__filterDiaposones':[[[8,12],[12,30]],
                                          [[6,10],[8,12],[10,14],[12,16],[14,18],[16,20],[18,22],[20,24]],
                                          [[4,8],[8,12],[12,24],[24,30]]],
               'cropper__startTime': [0, 1],
               'cropper__seriesLength': [1, 2, 3],
               'channels__selectedCnls': [["O2", "O1", "P4", "P3", "C4", "C3", "F4", "F3", "Fp2", "Fp1", "T6", "T5",
                                           "T4", "T3", "F8", "F7", "Pz", "Cz", "Fz"],
                   ["P4", "P3", "C4", "C3", "F4", "F3", "Fp2", "Fp1", "T6", "T5", "T4", "T3", "F8", "F7", "Pz", "Cz",
                    "Fz"]]

    }

    # ["P4", "P3", "C4", "C3", "F4", "F3", "Pz", "Cz", "Fz"]
    # filter=Bandpass(EEG.Frequency,10,20,axis=1)
    cropper=CropSeries(EEG.Frequency,1,3)
    # csp = CSP(n_components=4, reg=None, log=True)
    lda=LinearDiscriminantAnalysis()
    tr=Transpose()
    channels=SelectChannels(defaultCnls,defaultCnls)

    # tr1 = Transpose()
    svm = svm.SVC(kernel='rbf', C=1)
    FBCSP=FBCSP(n_components=5,filterDiaposones=[[6,10],[8,12],[10,14],[12,16],[14,18],[16,20],[18,22],[20,24]], frequency=EEG.Frequency,transform_into='max_power')
    pipe=Pipeline([('channels',channels),('cropper',cropper),('tr',tr),('FBCSP',FBCSP),('svm',lda)])
    # gs=GridSearchCV(pipe,gs_params,cv=5)
    # gs.fit(EEG.EEGData, EEG.Labels)
    # res=pipe.fit_transform(EEG.EEGData,EEG.Labels)
    # print(res)
    #
    # # union = Pipeline([('filter', filter),('cropper',cropper),('tr',tr),('csp',csp)])
    # # filtered=union.fit_transform(EEG.EEGData,EEG.Labels)
    # # print(filtered.shape)
    # # plt.plot( filtered[0,:,0])
    # #
    # # plt.show()
    # union=Pipeline([('filter', filter),('cropper',cropper),('tr',tr),('csp',csp),('lda',svm)])
    # gs=GridSearchCV(union,gs_params,cv=2)
    # gs.fit(EEG.EEGData, EEG.Labels)
    scores = cross_val_score(pipe, EEG.EEGData, EEG.Labels, cv=5)
    print(scores)
    # print("Best parameters set found on development set:")
    # print()
    # print(gs.best_params_)
    # print()
    # print("Grid scores on development set:")
    # print()
    # means = gs.cv_results_['mean_test_score']
    # stds = gs.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()
