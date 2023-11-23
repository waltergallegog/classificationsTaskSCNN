import warnings
from scipy.io.wavfile import read
import numpy as np
from scipy.signal import lfilter
import math
import random
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', message='.*Chunk.*')


############################################################
# ##### Dataset standardization and data elaboration ##### #
############################################################
##### Dataset standardization for audio sample #####
class DataAudio:
    def __init__(self, sample, channels):
        self.name = sample.split('/')[-1].replace('.wav', '')

        fs, data = read(sample)

        if len(data.shape) > 1:
            data = data.sum(axis=1)

        self.data = data
        self.fs = fs

        print(f'Sample: {self.name}')
        print(f'Channels: {channels}')
        print(f'Frequency sampling: {self.fs} Hz')

        # The human hearable range is from 20 - 20K Hz.
        # We calculate the range as [20, 1/2 sampling freq]
        self.freqRange = (20, np.floor(self.fs/2))
        print(f'Frequency range: \n{self.freqRange} Hz')
        print()

        freqMin, freqMax = self.freqRange
        octave = (channels-0.5)*np.log10(2)/np.log10(freqMax/freqMin)
        print(f'Octave: \n{octave}')
        print()

        # Equally spaced center frequencies in logarithmic scale
        self.freqCentr = np.array([freqMin*(2**(ch/octave)) for ch in range(channels)])
        print(f'Center frequencies: \n{self.freqCentr}')
        print()

        # width of bands that guarantee the minimum loss in gain of -3dB (0.7 in the plot)
        self.freqPoles = np.array([(freq*(2**(-1/(2*octave))), (freq*(2**(1/(2*octave))))) for freq in self.freqCentr])

        # The stop freq of filter 1 is equal to the start freq of filter 2 and so on
        self.freqPoles[-1, 1] = fs/2*0.99999
        print(f'Start and stop frequencies \n{self.freqPoles}')
        print()

    def decomposition(self, filterbank):
        self.components = []
        for num, den in filterbank:
            self.components.append(lfilter(num, den, self.data))


##### Dataset standardization for device sample #####
class DataDevice:
    def __init__(self, sample, fs, channels):
        self.data = []

        for key in ['accel', 'gyro']:
            [self.data.append(axis) for axis in sample[key]]

        self.fs = fs
        self.freqRange = (0.5, np.floor(self.fs/2))
        print(f'Frequency range: {self.freqRange} Hz')

        freqMin, freqMax = self.freqRange
        octave = (channels-0.5)*np.log10(2)/np.log10(freqMax/freqMin)
        self.freqCentr = np.array([freqMin*(2**(ch/octave)) for ch in range(channels)])
        self.freqPoles = np.array([(freq*(2**(-1/(2*octave))), (freq*(2**(1/(2*octave))))) for freq in self.freqCentr])
        self.freqPoles[-1, 1] = fs/2*0.99999

    def decomposition(self, filterbank):
        self.components = []
        for dataAxis in self.data:
            # tmp = []
            for num, den in filterbank:
                # tmp.append(lfilter(num, den, dataAxis))
                self.components.append(lfilter(num, den, dataAxis))
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
            tmp = np.array([])
            for sample in range(samples):
                tmp = np.concatenate((tmp, self.trainSet[sample].spikeTrain[cell]+((timeStimulus['duration']+timeStimulus['silence'])*np.array(sample))))
            self.spikeTrainSet.append(tmp)

        # creating a unique variable for spike train with source from test set
        self.spikeTestSet = []
        samples = dataset['testSet'][1].shape[0]
        for cell in range(cells):
            tmp = np.array([])
            for sample in range(samples):
                tmp = np.concatenate((tmp, self.testSet[sample].spikeTrain[cell]+((timeStimulus['duration']+timeStimulus['silence'])*np.array(sample))))
            self.spikeTestSet.append(tmp)


##### Dataset Rate encoding for SCNN #####
class SampleEncoding:
    def __init__(self, sample, timeStimulus, encoding):

        # separation data and label
        self.data = sample[0]  # image
        self.label = sample[1]  # label

        # spike generation
        if encoding == 'poisson':
            random.seed(0)

            self.spikeTrain = []

            rows, cols = self.data.shape
            for row in range(rows):
                for col in range(cols):
                    rate = self.data[row, col]  # intensity data in cell
                    if rate == 0:
                        self.spikeTrain.append(np.array([]))  # no stimulus
                    else:
                        spikeSequence = []
                        poissonISI = -math.log(1.0-random.random())/rate*1000.0  # ms tau
                        spikeTime = poissonISI
                        while spikeTime < timeStimulus['duration']:
                            spikeSequence.append(spikeTime)
                            poissonISI = -math.log(1.0-random.random())/rate*1000.0  # ms tau
                            spikeTime += poissonISI
                        self.spikeTrain.append(np.array(spikeSequence))

    ##### Plot sample #####
    def plotSample(self):
        plt.title('TBR')
        plt.imshow(self.data[:, :], cmap='viridis')
        plt.xlabel('Bins')
        plt.ylabel('Channels')
        plt.show()

    ##### Plot spike train #####
    def plotSpikeTrain(self):
        plt.title('TBR')
        plt.eventplot(self.spikeTrain)
        plt.xlabel('Time')
        plt.ylabel('Pixel')
        plt.show()
