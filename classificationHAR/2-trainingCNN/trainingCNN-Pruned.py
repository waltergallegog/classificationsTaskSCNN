import sys
sys.path.append('../../')
import argparse
from utils.graphicalUtility import *
from utils.netUtility import *
from utils.architecture import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


#############################
# ##### Training loop ##### #
#############################
def main(datasetName, encoding, filterbank, channel, bins, structure, quartile):
    ##### Dataset loading #####
    sourceFolder = '../../datasets/HumanActivityRecognition/datasetSonogram/'
    fileName = f'{sourceFolder}sonogram{datasetName}{filterbank}{channel}x{bins}{encoding}.bin'
    trainData, trainLabel, testData, testLabel, datasetClass = datasetSplitting(fileName, 'CNN')

    ##### Load model network #####
    sourceFolder = '../../networkModels/HumanActivityRecognition/complete/'
    modelCNN = tf.keras.models.load_model(f'{sourceFolder}{datasetName}{filterbank}{channel}x{bins}{structure}{encoding}.keras', custom_objects={'Relu': Relu})
    layersWeigths = modelCNN.get_weights()
    mask = modelCNN.get_weights()

    ##### Quartile calculation #####
    for i in range(len(layersWeigths)):
        boxplot_data = plt.boxplot(np.abs(layersWeigths[i].flatten()))
        threshold = 0
        if quartile == 'median':
            threshold = boxplot_data['medians'][0].get_ydata()[1]
        elif quartile == 'upper':
            threshold = boxplot_data['boxes'][0].get_ydata()[2]
        mask[i] = np.where((np.abs(layersWeigths[i]) < threshold), 0.0, 1.0)
        layersWeigths[i] = np.where((np.abs(layersWeigths[i]) < threshold), 0.0, layersWeigths[i])

    ##### Model definitions #####
    synapses = np.sum([np.sum(m) for m in mask], dtype=int)
    dataShape = trainData.shape[1:]
    modelCNNPruned = netModelsPruned(structure, dataShape, mask, datasetClass)

    # model summary
    # modelCNNPruned.summary()
    modelCNNPruned.set_weights(layersWeigths)
    modelCNNPruned.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    accuracyTrain = modelCNNPruned.fit(x=trainData, y=trainLabel, validation_split=0.1, epochs=10, batch_size=1, verbose=0)
    accuracyTest = modelCNNPruned.evaluate(x=testData, y=testLabel, verbose=0)

    return accuracyTrain.history['accuracy'][-1], accuracyTest[-1], synapses, modelCNNPruned


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-n', '--datasetName', help='Dataset file name', type=str, default='subset1')
    parser.add_argument('-e', '--encoding', help='Encoding algorithm selected', type=str, default='TBR')
    parser.add_argument('-f', '--filterbank', help='Type of filterbank', type=str, default='butterworth')
    parser.add_argument('-c', '--channel', help='Frequency decomposition channels', type=int, default=4)
    parser.add_argument('-b', '--bins', help='Binning width', type=int, default=24)
    parser.add_argument('-s', '--structure', help='Network structure', type=str, default='c06c12f2')
    parser.add_argument('-q', '--quartile', help='Quartile pruning', type=str, default='median')
    parser.add_argument('-t', '--trials', help='Trials training', type=int, default=20)

    argument = parser.parse_args()

    datasetName = argument.datasetName
    encoding = argument.encoding
    filterbank = argument.filterbank
    channel = argument.channel
    bins = argument.bins
    structure = argument.structure
    quartile = argument.quartile
    trials = argument.trials


    ##### Check model already calculated #####
    columnLabels = ['Filterbank', 'Channels', 'Bins', 'Encoding', 'Structure', 'Quartile', 'Synapses', 'Train', 'Test']
    flagCompute = True
    sourceFolder = '../../networkPerformance/HumanActivityRecognition/'
    fileName = f'{sourceFolder}{datasetName}CNN-ModelPruned.csv'
    try:
        performanceData = pd.read_csv(fileName)
        flagCompute = not bool(len(performanceData[
            (performanceData['Encoding'] == encoding) &
            (performanceData['Filterbank'] == filterbank) &
            (performanceData['Channels'] == channel) &
            (performanceData['Bins'] == bins) &
            (performanceData['Structure'] == structure) &
            (performanceData['Quartile'] == quartile)
        ]))
    except:
        pass

    ##### Run training models #####
    print(datasetName, encoding, filterbank, channel, bins, structure, quartile)
    if flagCompute == True:
        metrics = np.zeros(trials)
        history = []
        progressBar(0, trials)
        for trial in range(trials):
            accuracyTrain, accuracyTest, synapses, modelCNNPruned = main(datasetName, encoding, filterbank, channel, bins, structure, quartile)
            history.append((accuracyTrain, accuracyTest, synapses, modelCNNPruned))
            metrics[trial] = (1-accuracyTrain)**2+(1-accuracyTest)**2
            progressBar(trial+1, trials)

        accuracyTrain, accuracyTest, synapses, modelCNNPruned = history[np.argmin(metrics)]

        ##### Save data of models #####
        sourceFolder = '../../networkModels/HumanActivityRecognition/pruned/'
        modelCNNPruned.save(f'{sourceFolder}{datasetName}{filterbank}{channel}x{bins}{structure}{quartile}{encoding}.keras')

        ##### Save data for performance #####
        try:
            performanceData = pd.read_csv(fileName)
            performanceData = performanceData.values.tolist()
            performanceData.append([filterbank, channel, bins, encoding, structure, quartile, synapses, accuracyTrain, accuracyTest])
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
        except:
            performanceData = [[filterbank, channel, bins, encoding, structure, quartile, synapses, accuracyTrain, accuracyTest]]
            performanceData = pd.DataFrame(performanceData, index=None, columns=columnLabels)
            performanceData.to_csv(fileName, index=False)
