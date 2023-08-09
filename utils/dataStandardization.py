import warnings as warningsLocal
from scipy.io.wavfile import read as readLocal
import numpy as npLocal
from scipy.signal import lfilter as lfilterLocal
import math as mathLocal
import random as randomLocal
import matplotlib.pyplot as pltLocal

warningsLocal.filterwarnings('ignore', message='.*Chunk.*')


############################################################
# ##### Dataset standardization and data elaboration ##### #
############################################################
##### Dataset standardization for audio sample #####
class DataAudio:
    def __init__(self, sample, channels):
        self.name = sample.split('/')[-1].replace('.wav', '')

        fs, data = readLocal(sample)

        if len(data.shape) > 1:
            data = data.sum(axis=1)

        self.data = data
        self.fs = fs
        self.freqRange = (20, npLocal.floor(self.fs/2))

        freqMin, freqMax = self.freqRange
        octave = (channels-0.5)*npLocal.log10(2)/npLocal.log10(freqMax/freqMin)
        self.freqCentr = npLocal.array([freqMin*(2**(ch/octave)) for ch in range(channels)])
        self.freqPoles = npLocal.array([(freq*(2**(-1/(2*octave))), (freq*(2**(1/(2*octave))))) for freq in self.freqCentr])
        self.freqPoles[-1, 1] = fs/2*0.99999

    def decomposition(self, filterbank):
        self.components = []
        for num, den in filterbank:
            self.components.append(lfilterLocal(num, den, self.data))


##### Dataset standardization for device sample #####
class DataDevice:
    def __init__(self, sample, fs, channels):
        self.data = []

        for key in ['accel', 'gyro']:
            [self.data.append(axis) for axis in sample[key]]

        self.fs = fs
        self.freqRange = (0.5, npLocal.floor(self.fs/2))

        freqMin, freqMax = self.freqRange
        octave = (channels-0.5)*npLocal.log10(2)/npLocal.log10(freqMax/freqMin)
        self.freqCentr = npLocal.array([freqMin*(2**(ch/octave)) for ch in range(channels)])
        self.freqPoles = npLocal.array([(freq*(2**(-1/(2*octave))), (freq*(2**(1/(2*octave))))) for freq in self.freqCentr])
        self.freqPoles[-1, 1] = fs/2*0.99999

    def decomposition(self, filterbank):
        self.components = []
        for dataAxis in self.data:
            # tmp = []
            for num, den in filterbank:
                # tmp.append(lfilterLocal(num, den, dataAxis))
                self.components.append(lfilterLocal(num, den, dataAxis))
            # self.components.append(tmp)


##### Dataset Rate encoding for SCNN #####
class Dataset:
    def __init__(self, dataset, timeStimulus, encoding):

        # self.train_set = [SampleEncoding(element, time_stimulus, spike_train_model) for element in dataset['train_set']]
        # self.test_set = [SampleEncoding(element, time_stimulus, spike_train_model) for element in dataset['test_set']]
        self.trainSet = []
        for i in range(dataset['trainSet'][1].shape[0]):
            sample = dataset['trainSet'][0][i], dataset['trainSet'][1][1]
            self.trainSet.append(SampleEncoding(sample, timeStimulus, encoding))
        self.testSet = []
        for i in range(dataset['testSet'][1].shape[0]):
            sample = dataset['testSet'][0][i], dataset['testSet'][1][1]
            self.testSet.append(SampleEncoding(sample, timeStimulus, encoding))

        # number of cells in the input data
        self.shape = dataset['trainSet'][0][0].shape
        cells = self.shape[0]*self.shape[1]

        # creating a unique variable for spike train with source from training set
        self.spikeTrainSet = []
        samples = dataset['trainSet'][1].shape[0]
        for cell in range(cells):
            tmp = npLocal.array([])
            for sample in range(samples):
                tmp = npLocal.concatenate((tmp, self.trainSet[sample].spikeTrain[cell]+((timeStimulus['duration']+timeStimulus['silence'])*npLocal.array(sample))))
            self.spikeTrainSet.append(tmp)

        # creating a unique variable for spike train with source from test set
        self.spikeTestSet = []
        samples = dataset['testSet'][1].shape[0]
        for cell in range(cells):
            tmp = npLocal.array([])
            for sample in range(samples):
                tmp = npLocal.concatenate((tmp, self.testSet[sample].spikeTrain[cell]+((timeStimulus['duration']+timeStimulus['silence'])*npLocal.array(sample))))
            self.spikeTestSet.append(tmp)


##### Dataset Rate encoding for SCNN #####
class SampleEncoding:
    def __init__(self, sample, timeStimulus, encoding):

        # separation data and label
        self.data = sample[0]  # image
        self.label = sample[1]  # label

        # spike generation
        if encoding == 'poisson':
            randomLocal.seed(0)

            self.spikeTrain = []

            rows, cols = self.data.shape
            for row in range(rows):
                for col in range(cols):
                    rate = self.data[row, col]  # intensity data in cell
                    if rate == 0:
                        self.spikeTrain.append(npLocal.array([]))  # no stimulus
                    else:
                        spikeSequence = []
                        poissonISI = -mathLocal.log(1.0-randomLocal.random())/rate*1000.0  # ms tau
                        spikeTime = poissonISI
                        while spikeTime < timeStimulus['duration']:
                            spikeSequence.append(spikeTime)
                            poissonISI = -mathLocal.log(1.0-randomLocal.random())/rate*1000.0  # ms tau
                            spikeTime += poissonISI
                        self.spikeTrain.append(npLocal.array(spikeSequence))

    ##### Plot sample #####
    def plotSample(self):
        pltLocal.title('TBR')
        pltLocal.imshow(self.data[:, :], cmap='viridis')
        pltLocal.xlabel('Bins')
        pltLocal.ylabel('Channels')
        pltLocal.show()

    ##### Plot spike train #####
    def plotSpikeTrain(self):
        pltLocal.title('TBR')
        pltLocal.eventplot(self.spikeTrain)
        pltLocal.xlabel('Time')
        pltLocal.ylabel('Pixel')
        pltLocal.show()
