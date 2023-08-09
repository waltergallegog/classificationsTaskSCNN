import sys
sys.path.append('../../')
import argparse
from utils.graphicalUtility import *
from utils.dataStandardization import *
import numpy as np
from scipy.signal import butter, gammatone
from scipy.signal.windows import *
from utils.dataEncoding import *
import pickle


def main(datasetDevice, datasetName, subsetLabel, filterbank, channel):
    encodings = ['RATE', 'TBR', 'SF', 'MW', 'PFM', 'HSA', 'MHSA', 'BSA', 'PHASE', 'TTFS', 'BURST']

    sourceFolder = f'../../datasets/HumanActivityRecognition/datasetRaw/'
    file = open(f'{sourceFolder}{datasetDevice}.bin', 'rb')
    dataset = pickle.load(file)
    file.close()

    subset = [dataset[label] for label in subsetLabel]
    sampleNum = len(subset[0])
    ##### Memory for encoded data #####
    spikeset = [[] for _ in range(len(encodings))]
    progressBar(0, len(subsetLabel)*sampleNum)
    for index_class in range(len(subsetLabel)):
        for index_sample in range(sampleNum):
            ##### Data standardization #####
            # print(subsetLabel[index_class], index_sample, end=' ')
            progressBar((index_sample+1)+(index_class*sampleNum), len(subsetLabel)*sampleNum, f'{datasetName}, {filterbank}, {channel}')
            sample = DataDevice(subset[index_class][index_sample], 20, channel)
            for i in range(len(sample.data)):
                padding = int((4802-len(sample.data[i]))/2)
                padding = np.where(padding > 0, padding, 0)
                sample.data[i] = np.pad(sample.data[i], padding, 'constant')[:4800]

            ##### Frequency decomposition #####
            if filterbank == 'butterworth':
                order = 2
                butter_filterbank = []
                for fl, fh in sample.freqPoles:
                    num, den = butter(N=order, Wn=(fl, fh), btype='band', fs=sample.fs)
                    butter_filterbank.append([num, den])
                sample.decomposition(butter_filterbank)
            elif filterbank == 'gammatone':
                order = 1
                gammatone_filterbank = []
                for f in sample.freqCentr:
                    num, den = gammatone(order=order, freq=f, ftype='fir', fs=sample.fs)
                    gammatone_filterbank.append([num, den])
                sample.decomposition(gammatone_filterbank)

            #####################################
            # ##### Parameters definition ##### #
            #####################################
            threshold_sf = np.mean([component.max()-component.min() for component in sample.components])/10
            threshold_mw = [np.mean(np.abs(component[1:]-component[:-1])) for component in sample.components]
            threshold_pfm = np.mean([component.max()-component.min() for component in sample.components])/10
            filters = boxcar(3)
            setting = {
                'tbr_factors': 0.5,
                'sf_thresholds': threshold_sf,
                'mw_window': 3, 'mw_thresholds': threshold_mw,
                'pfm_threshold': threshold_pfm,
                'hsa_filter': filters,
                'mhsa_filter': filters, 'mhsa_threshold': 0.85,
                'bsa_filter': filters, 'bsa_threshold': 1,
                'phase_bit': 6,
                'ttfs_interval': 10,
                'N_max': 5, 't_min': 0, 't_max': 4, 'burst_length': 13
            }

            #################################
            # ##### Encoding settings ##### #
            #################################
            ##### Rate coding #####
            encoding_RC = RateCoding(sample)
            encoding_RC.RATE(setting)

            ##### Temporal Contrast #####
            encoding_TC = TemporalContrast(sample)
            encoding_TC.TBR(setting)
            encoding_TC.SF(setting)
            encoding_TC.MW(setting)
            encoding_TC.PFM(setting)

            ##### Filter Optimizer #####
            encoding_FO = FilterOptimizer(sample)
            encoding_FO.HSA(setting)
            encoding_FO.MHSA(setting)
            encoding_FO.BSA(setting)

            ##### Global Referenced #####
            encoding_GR = GlobalReferenced(sample)
            encoding_GR.PHASE(setting)
            encoding_GR.TTFS(setting)
            encoding_GR.BURST(setting)

            ################################
            # ##### Save data on RAM ##### #
            ################################
            ##### Rate coding #####
            spikeset[0].append([encoding_RC.RATE_aer.addresses, encoding_RC.RATE_aer.timestamps, index_class])

            ##### Temporal Contrast #####
            spikeset[1].append([encoding_TC.TBR_aer.addresses, encoding_TC.TBR_aer.timestamps, index_class])
            spikeset[2].append([encoding_TC.SF_aer.addresses, encoding_TC.SF_aer.timestamps, index_class])
            spikeset[3].append([encoding_TC.MW_aer.addresses, encoding_TC.MW_aer.timestamps, index_class])
            spikeset[4].append([encoding_TC.PFM_aer.addresses, encoding_TC.PFM_aer.timestamps, index_class])

            ##### Filter Optimizer #####
            spikeset[5].append([encoding_FO.HSA_aer.addresses, encoding_FO.HSA_aer.timestamps, index_class])
            spikeset[6].append([encoding_FO.MHSA_aer.addresses, encoding_FO.MHSA_aer.timestamps, index_class])
            spikeset[7].append([encoding_FO.BSA_aer.addresses, encoding_FO.BSA_aer.timestamps, index_class])

            ##### Global Referenced #####
            spikeset[8].append([encoding_GR.PHASE_aer.addresses, encoding_GR.PHASE_aer.timestamps, index_class])
            spikeset[9].append([encoding_GR.TTFS_aer.addresses, encoding_GR.TTFS_aer.timestamps, index_class])
            spikeset[10].append([encoding_GR.BURST_aer.addresses, encoding_GR.BURST_aer.timestamps, index_class])
    
    ##### Save data on numpy file #####
    sourceFolder = f'../../datasets/HumanActivityRecognition/datasetSpike/'
    for i, encoding in enumerate(encodings):
        file = open(f'{sourceFolder}spikeset{datasetName}{filterbank}{channel}{encoding}.bin', 'wb')
        pickle.dump(spikeset[i], file)
        file.close()

    return 0


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-d', '--datasetDevice', help='Device dataset', type=str, default='datasetWatch')
    parser.add_argument('-n', '--datasetName', help='Dataset file name', type=str, default='subset1')
    parser.add_argument('-s', '--subsetLabel', help='List of subset class', type=str, default='A,B,G,H,P,R')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channel', help='Frequency decomposition channels', type=int, default=4)

    argument = parser.parse_args()

    ##### Parsing unpack #####
    datasetDevice = argument.datasetDevice
    datasetName = argument.datasetName
    subsetLabel = argument.subsetLabel.split(',')
    filterbank = argument.filterbank
    channel = argument.channel

    main(datasetDevice, datasetName, subsetLabel, filterbank, channel)
