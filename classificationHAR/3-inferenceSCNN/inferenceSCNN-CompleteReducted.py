import sys
sys.path.append('../../')
import argparse
from utils.netUtility import *
from utils.dataStandardization import *
import tensorflow as tf
from utils.architecture import *
from utils.conversionCNN import *
from utils.conversionSCNN import *
import numpy as np
import pandas as pd


def spikeLabeling(datasetClass, trueLabel, time, spikeTrainNest):
    ##### Conversion spike train from NEO to Numpy format #####
    numberSamples = trueLabel.shape[0]
    spikeTrain = []
    for layer in spikeTrainNest:
        tmp = []
        for neuron in layer:
            tmp.append(np.array(neuron))
        spikeTrain.append(tmp)

    ##### Extraction and classification and accuracy calculation #####
    spikeCount = np.zeros((datasetClass, numberSamples), dtype=int)
    intervalTime = [time*i for i in range(1, numberSamples)]
    for i in range(datasetClass):
        classNeuron = spikeTrain[-1][i]
        for t in np.searchsorted(intervalTime, classNeuron, side='right'):
            spikeCount[i, t] += 1
    prediction = np.argmax(spikeCount, axis=0)
    accuracy = np.sum(prediction == trueLabel)/numberSamples

    return accuracy


##############################
# ##### Inference loop ##### #
##############################
def main(datasetName, encoding, filterbank, channel, bins, structure, reduction):
    ###############################
    # ##### Dataset loading ##### #
    ###############################
    sourceFolder = '../../datasets/HumanActivityRecognition/datasetSonogram/'
    fileName = f'{sourceFolder}sonogram{datasetName}{filterbank}{channel}x{bins}{encoding}.bin'
    trainData, trainLabel, testData, testLabel, datasetClass = datasetSplitting(fileName, 'SNN')

    timeStimulus = {'duration': 1000.0, 'silence': 20.0}

    dataset = Dataset(
        {'trainSet': (trainData, trainLabel), 'testSet': (testData, testLabel)},
        timeStimulus,
        'poisson'
    )

    # spokenDigit.testSet[0].plotSample()
    # spokenDigit.testSet[0].plotSpikeTrain()

    ##########################################
    # ##### Load pre-trained CNN model ##### #
    ##########################################
    # loading data from backup weigths
    sourceFolder = '../../networkModels/HumanActivityRecognition/complete/'
    modelCNN = tf.keras.models.load_model(f'{sourceFolder}{datasetName}{filterbank}{channel}x{bins}{structure}{encoding}.keras', custom_objects={'Relu': Relu})

    # data structuring for SNN conversion
    modelCNN = CNN(modelCNN)

    ###################################
    # ##### Simulation with SNN ##### #
    ###################################
    reductionCode = [int(i) for i in reduction]
    # parameter of neuron LIF
    lifParams = {
        'cm': 0.25,  # nF
        'i_offset': 0.1,  # nA
        'tau_m': 20.0,  # ms
        'tau_refrac': 1.0,  # ms
        'tau_syn_E': 5.0,  # ms
        'tau_syn_I': 5.0,  # ms
        'v_reset': -65.0,  # mV
        'v_rest': -65.0,  # mV
        'v_thresh': -50.0  # mV
    }
    time = timeStimulus['duration']+timeStimulus['silence']
    shapeData = dataset.shape

    modelSNN = SNN(shapeData, dataset.spikeTrainSet, modelCNN, lifParams, reductionCode)

    numberSamples = len(dataset.trainSet)
    spikeTrainNest = modelSNN.start_simulation(numberSamples, timeStimulus)
    accuracyTrain = spikeLabeling(datasetClass, trainLabel, time, spikeTrainNest)

    modelSNN = SNN(shapeData, dataset.spikeTestSet, modelCNN, lifParams, reductionCode)

    numberSamples = len(dataset.testSet)
    spikeTestNest = modelSNN.start_simulation(numberSamples, timeStimulus)
    accuracyTest = spikeLabeling(datasetClass, testLabel, time, spikeTestNest)

    return accuracyTrain, accuracyTest, modelSNN.synapses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-n', '--datasetName', help='Dataset file name', type=str, default='subset1')
    parser.add_argument('-e', '--encoding', help='Encoding algorithm selected', type=str, default='RATE')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channel', help='Frequency decomposition channels', type=int, default=4)
    parser.add_argument('-b', '--bins', help='Binning width', type=int, default=24)
    parser.add_argument('-s', '--structure', help='Network structure', type=str, default='c06c12f2')
    parser.add_argument('-r', '--reduction', help='Synaptic reduction based on the weights', type=str, default='000000')

    argument = parser.parse_args()

    datasetName = argument.datasetName
    encoding = argument.encoding
    filterbank = argument.filterbank
    channel = argument.channel
    bins = argument.bins
    structure = argument.structure
    reduction = argument.reduction

    ##### Check model already calculated #####
    columnLabels = ['Filterbank', 'Channels', 'Bins', 'Encoding', 'Structure', 'Reduction', 'Synapses', 'Train', 'Test']
    flagCompute = True
    sourceFolder = '../../networkPerformance/HumanActivityRecognition/'
    fileName = f'{sourceFolder}{datasetName}SCNN-ModelCompleteReduced.csv'
    try:
        performanceData = pd.read_csv(fileName)
        flagCompute = not bool(len(performanceData[
            (performanceData['Encoding'] == encoding) &
            (performanceData['Filterbank'] == filterbank) &
            (performanceData['Channels'] == channel) &
            (performanceData['Bins'] == bins) &
            (performanceData['Structure'] == structure) &
            (performanceData['Reduction'] == int(reduction))
        ]))
    except:
        pass

    print(encoding, filterbank, channel, bins, structure, reduction)
    if flagCompute == True:
        accuracyTrain, accuracyTest, synapses = main(datasetName, encoding, filterbank, channel, bins, structure, reduction)

        ##### Save data for performance #####
        try:
            performanceData = pd.read_csv(fileName)
            performanceData = performanceData.values.tolist()
            performanceData.append([filterbank, channel, bins, encoding, structure, reduction, synapses, accuracyTrain, accuracyTest])
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
        except:
            performanceData = [[filterbank, channel, bins, encoding, structure, reduction, synapses, accuracyTrain, accuracyTest]]
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
