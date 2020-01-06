import numpy as np
import json
import os
class EEGData:
    # Frequency - частота дискретизации
    # ChannelCount - количество каналов
    # ActivityTime - длина одной попытки в секундах
    # ActivityTypes - названия классифицуремых действий в сессии
    # TrialsCount - количество попыток для каждого действия
    # EEGData - трехмерный массив вида arr[a,b,c],гдe
    #           a - общее количество попыток по всем классам в сессиии = sum(TrialsCount)
    #           b - количество отсчетов в попытке = Frequency*ActivityTime
    #           c - количество каналов
    # Labels - одномерный массив размера sum(TrialsCount) показывающий принадлежность попытки к классу

    def __init__(self):
        self.GetData()
    def GetData(self):

        jsonSeetings = open(os.path.join(os.path.dirname(__file__), 'Settings.json'))
        settingsData = json.load(jsonSeetings)
        self.Path = settingsData['rootFolder'] + '\\' + settingsData['sessionFolder']
        jsonSession = open(self.Path + '\\Session params.json')
        sessionData = json.load(jsonSession)

        self.Frequency = int(sessionData['frequency'])
        self.ChannelCount = int(sessionData['channelCount'])
        #self.RestTime = int(params['restTime']) #пока не нужен, все равно читаем резанные файлы без этого параметра
        self.ActivityTime = int(sessionData['activityTime'])
        self.ActivityTypes = [activity for activity in sessionData['activity']]
        self.TrialsCount = [int(i) for i in sessionData['trials']]
        self.ChannelNames=[i for i in sessionData['channelNames']]

        self.GetEEG()

    #Получаем ЭЭГ из файлов с разбитыми данными
    def GetEEG(self):
        self.EEGData = np.zeros(shape=(sum(self.TrialsCount), self.Frequency * self.ActivityTime, self.ChannelCount), dtype=float)
        self.Labels = np.zeros(sum(self.TrialsCount))
        try:
            trial = 0
            for file in os.listdir(os.path.join(self.Path, "Trials")):
                if os.path.splitext(file)[1] == ".csv":
                    label = self.ActivityTypes.index(os.path.splitext(file)[0])
                    data = open(os.path.join(self.Path, "Trials", file))
                    i = 0
                    for line in data:
                        self.EEGData[trial, i, :] = np.array((line.split(';'))).astype(float)[0:self.ChannelCount]
                        i += 1
                        if i == self.Frequency * self.ActivityTime:
                            i = 0
                            self.Labels[trial] = label
                            trial += 1

        except FileNotFoundError:
            print("Сеcсия не разбита на попытки")
            exit()

    def Transpose(self):
        bufData=np.zeros(shape=(self.EEGData.shape[0],self.EEGData.shape[2],self.EEGData.shape[1]),dtype=float)
        for i in range(self.EEGData.shape[0]):
            bufData[i,:,:]=np.matrix.transpose(self.EEGData[i,:,:])
        self.EEGData=bufData

class Channels:
    def __init__(self, electrodes):
        self._electrodes = electrodes
if __name__=="__main__":
    pass