import sys
sys.path.append('../../')
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np


def featureExtraction(spikeSet, channel, binsWindow, plot=False):
    tmp = []
    binsSample = int(8000*binsWindow/1000)
    binsNumber = int(1000/binsWindow)
    for indexSample in range(len(spikeSet)):
        spikeTrain = np.zeros((channel, 8000))
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
            plt.imshow(sonogram, aspect='auto', vmax=75, vmin=0)
            plt.show()
        tmp.append([sonogram, label])
    return binsNumber, tmp


def main(encoding, filterbank, channel, binsWindow):
    ##################################
    # ##### Load dataset spike ##### #
    ##################################
    sourceFolder = '../../datasets/FreeSpokenDigits/datasetSpike/'
    file = open(f'{sourceFolder}spikeset{filterbank}{channel}{encoding}.bin', 'rb')
    spikeSet = pickle.load(file)
    file.close()

    #################################
    # ##### Sonogram creation ##### #
    #################################
    bins, dataset = featureExtraction(spikeSet, channel, binsWindow)

    #####################################
    # ##### Save sonogram dataset ##### #
    #####################################
    sourceFolder = f'../../datasets/FreeSpokenDigits/datasetSonogram/'
    file = open(f'{sourceFolder}sonogram{filterbank}{channel}x{bins}{encoding}.bin', 'wb')
    pickle.dump(dataset, file)
    file.close()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-e', '--encoding', help='Encoding algorithm selected', type=str, default='RATE')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channel', help='Frequency decomposition channels', type=int, default=32)
    parser.add_argument('-b', '--binsWindow', help='Binning width', type=float, default=20.0)

    argument = parser.parse_args()

    ##### Parsing unpack #####
    encoding = argument.encoding
    filterbank = argument.filterbank
    channel = argument.channel
    binsWindow = argument.binsWindow

    main(encoding, filterbank, channel, binsWindow)
