import sys
sys.path.append('../../')
import argparse
from utils import progressBar
from utils import datasetSplitting
from utils import netModelsComplete
import numpy as np
import pandas as pd


#############################
# ##### Training loop ##### #
#############################
def main(encoding, filterbank, channel, bins, structure):
    ##### Dataset loading #####
    sourceFolder = '../../datasets/FreeSpokenDigits/datasetSonogram/'
    fileName = f'{sourceFolder}sonogram{filterbank}{channel}x{bins}{encoding}.bin'
    trainData, trainLabel, testData, testLabel, datasetClass = datasetSplitting(fileName, 'CNN')

    ##### Model definitions #####
    dataShape = trainData.shape[1:]
    modelCNN = netModelsComplete(structure, dataShape, datasetClass)

    # model summary
    # modelCNN.summary()

    modelCNN.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    accuracyTrain = modelCNN.fit(x=trainData, y=trainLabel, validation_split=0.1, epochs=30, batch_size=1, verbose=0)
    accuracyTest = modelCNN.evaluate(x=testData, y=testLabel, verbose=0)

    return accuracyTrain.history['accuracy'][-1], accuracyTest[-1], modelCNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-e', '--encoding', help='Encoding algorithm selected', type=str, default='RATE')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channel', help='Frequency decomposition channels', type=int, default=32)
    parser.add_argument('-b', '--bins', help='Binning width', type=int, default=50)
    parser.add_argument('-s', '--structure', help='Network structure', type=str, default='c06c12f2')
    parser.add_argument('-t', '--trials', help='Trials training', type=int, default=20)

    argument = parser.parse_args()

    encoding = argument.encoding
    filterbank = argument.filterbank
    channel = argument.channel
    bins = argument.bins
    structure = argument.structure
    trials = argument.trials

    ##### Check model already calculated #####
    columnLabels = ['Filterbank', 'Channels', 'Bins', 'Encoding', 'Structure', 'Train', 'Test']
    flagCompute = True
    sourceFolder = '../../networkPerformance/FreeSpokenDigits/'
    fileName = f'{sourceFolder}CNN-ModelComplete.csv'
    try:
        performanceData = pd.read_csv(fileName)
        flagCompute = not bool(len(performanceData[
            (performanceData['Encoding'] == encoding) &
            (performanceData['Filterbank'] == filterbank) &
            (performanceData['Channels'] == channel) &
            (performanceData['Bins'] == bins) &
            (performanceData['Structure'] == structure)
        ]))
    except:
        pass

    ##### Run training models #####
    print(encoding, filterbank, channel, bins, structure)

    if flagCompute == True:
        metrics = np.zeros(trials)
        history = []
        progressBar(0, trials)
        for trial in range(trials):
            accuracyTrain, accuracyTest, modelCNN = main(encoding, filterbank, channel, bins, structure)
            history.append((accuracyTrain, accuracyTest, modelCNN))
            metrics[trial] = (1-accuracyTrain)**2+(1-accuracyTest)**2
            progressBar(trial+1, trials)

        accuracyTrain, accuracyTest, modelCNN = history[np.argmin(metrics)]

        ##### Save data of models #####
        sourceFolder = '../../networkModels/FreeSpokenDigits/complete/'
        modelCNN.save(f'{sourceFolder}{filterbank}{channel}x{bins}{structure}{encoding}.keras')

        ##### Save data for performance #####
        try:
            performanceData = pd.read_csv(fileName)
            performanceData = performanceData.values.tolist()
            performanceData.append([filterbank, channel, bins, encoding, structure, accuracyTrain, accuracyTest])
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
        except:
            performanceData = [[filterbank, channel, bins, encoding, structure, accuracyTrain, accuracyTest]]
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
