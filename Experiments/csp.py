import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..','Modules'))
from eegtransformer import SelectChannels,CropSeries,Bandpass,Transpose,DownSample,CAR
from eegreader import EEGData
import numpy as np
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score,KFold
from sklearn import svm
import mne
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
#"O2", "O1", "P4", "P3", "C4", "C3", "F4", "F3", "Fp2", "Fp1", "T6", "T5", "T4", "T3", "F8", "F7", "Pz", "Cz", "Fz"
if __name__=='__main__':

    EEG=EEGData()

    defaultCnls=("O2", "O1", "P4", "P3", "C4", "C3", "F4", "F3", "Fp2", "Fp1", "T6", "T5", "T4", "T3", "F8", "F7", "Pz", "Cz", "Fz")
    gs_params={'csp__n_components':range(1,10),
               'filter__lowF':[2,6,7,8],
               'filter__highF':[12,24,30,40],
               'cropper__startTime':[0,1],
               'cropper__seriesLength':[1,2,3],
               'channels__selectedCnls':[[ "P4", "P3", "C4", "C3", "F4", "F3", "Fp2", "Fp1", "T6", "T5", "T4", "T3", "F8", "F7", "Pz", "Cz", "Fz"],
                                         ["P4", "P3", "C4", "C3", "F4", "F3","Pz", "Cz", "Fz"]]

    }
    channels=SelectChannels(defaultCnls,("C4", "C3", "F4"))
    filter=Bandpass(EEG.Frequency,8,12)
    cropper=CropSeries(EEG.Frequency,0,2)
    csp = CSP(n_components=4, reg=None, log=True)
    lda=LinearDiscriminantAnalysis()
    tr=Transpose()
    tr1 = Transpose()
    svm = svm.SVC(kernel='rbf', C=1)
    ds=DownSample(3)
    car=CAR()
    # union = Pipeline([('filter', filter),('cropper',cropper),('tr',tr),('csp',csp)])
    # filtered=union.fit_transform(EEG.EEGData,EEG.Labels)
    # print(filtered.shape)
    # plt.plot( filtered[0,:,0])
    #
    # plt.show()
    union=Pipeline([('tr',tr),('filter', filter),('ds',ds),('cropper',cropper),('csp',csp),('lda',svm)])

    # gs=GridSearchCV(union,gs_params,cv=3)
    # gs.fit(EEG.EEGData, EEG.Labels)

    # scores = cross_val_score(union, EEG.EEGData, EEG.Labels, cv=2)
    # print(scores)
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
    # print(car.transform(EEG.EEGData).shape)
    events=np.array([[EEG.Frequency*EEG.ActivityTime*i,0,EEG.Labels[i]] for i in range(len(EEG.Labels))],dtype=int)
    print(events)
    ch_types=['eeg' for i in range(EEG.ChannelCount)]
    ch_names=EEG.ChannelNames
    info=mne.create_info(ch_names=ch_names, sfreq=EEG.Frequency, ch_types=ch_types)
    montage = mne.channels.read_montage('standard_1020', ch_names=ch_names)
    epochs=mne.EpochsArray(np.square(Pipeline([('tr',tr),('filter',filter)]).transform(EEG.EEGData)),info=info, events=events)

    picks = mne.pick_types(info, meg=False, eeg=True, misc=False)
    epochs.plot(picks=picks, scalings='auto', show=True, block=True)
    epochs.set_montage(montage)