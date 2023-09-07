import sys
sys.path.append('../../')
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np


def featureExtraction(spikeSet, channel, binsWindow, plot=False):
    channel *= 6
    tmp = []
    binsSample = int(4800*binsWindow/1000)
    binsNumber = int(1000/binsWindow)
    for indexSample in range(len(spikeSet)):
        spikeTrain = np.zeros((channel, 4800))
        label = spikeSet[indexSample][2]
        for i in range(len(spikeSet[indexSample][0])):
            spikeTrain[spikeSet[indexSample][0][i], spikeSet[indexSample][1][i]] = 1
        sonogram = np.zeros((channel, binsNumber))
        for slices in range(binsNumber):
            sonogram[:, slices] = np.sum(spikeTrain[:, 0+slices*binsSample:binsSample+slices*binsSample], axis=1)
        if plot:
            plt.figure()
            plt.imshow(spikeTrain, aspect='auto')
            plt.figure()
            plt.xlabel('Bins')
            plt.ylabel('Channels')
            plt.imshow(sonogram, aspect='auto', vmax=85, vmin=0)
            plt.show()
        tmp.append([sonogram, label])
    return binsNumber, tmp


def main(datasetName, encoding, filterbank, channel, binsWindow):
    ##################################
    # ##### Load dataset spike ##### #
    ##################################
    sourceFolder = f'../../datasets/HumanActivityRecognition/datasetSpike/'
    file = open(f'{sourceFolder}spikeset{datasetName}{filterbank}{channel}{encoding}.bin', 'rb')
    spikeSet = pickle.load(file)
    file.close()

    #################################
    # ##### Sonogram creation ##### #
    #################################
    bins, dataset = featureExtraction(spikeSet, channel, binsWindow)

    #####################################
    # ##### Save sonogram dataset ##### #
    #####################################
    sourceFolder = f'../../datasets/HumanActivityRecognition/datasetSonogram/'
    file = open(f'{sourceFolder}sonogram{datasetName}{filterbank}{channel}x{bins}{encoding}.bin', 'wb')
    pickle.dump(dataset, file)
    file.close()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-n', '--datasetName', help='Dataset file name', type=str, default='subset1')
    parser.add_argument('-e', '--encoding', help='Encoding algorithm selected', type=str, default='RATE')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channel', help='Frequency decomposition channels', type=int, default=4)
    parser.add_argument('-b', '--binsWindow', help='Binning width', type=float, default=1000/24)

    argument = parser.parse_args()

    ##### Parsing unpack #####
    datasetName = argument.datasetName
    encoding = argument.encoding
    filterbank = argument.filterbank
    channel = argument.channel
    binsWindow = argument.binsWindow

    main(datasetName, encoding, filterbank, channel, binsWindow)
