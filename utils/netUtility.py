import pickle
from random import shuffle
import numpy as np

def datasetSplitting(fileName, netType):

    file = open(fileName, 'rb')
    sonogramSet = pickle.load(file)
    file.close()

    datasetClass = sonogramSet[-1][-1]+1

    classNum = sonogramSet[-1][-1]+1
    sampleNum = int(len(sonogramSet)/classNum)
    splitTrain, splitTest = int(np.floor(sampleNum*0.8)), int(np.ceil(sampleNum*0.2))

    ##### Data preprocessing #####
    trainSet = []
    testSet = []
    for digit in range(classNum):
        tmp = sonogramSet[0+sampleNum*digit:splitTest+sampleNum*digit]
        for sample in tmp:
            testSet.append(sample)
        tmp = sonogramSet[splitTest+sampleNum*digit:sampleNum+sampleNum*digit]
        for sample in tmp:
            trainSet.append(sample)
    shuffle(trainSet)
    shuffle(testSet)

    trainData = []
    trainLabel = []
    for data, label in trainSet:
        trainData.append([data])
        trainLabel.append(label)
    trainData = np.vstack(trainData)
    trainLabel = np.array(trainLabel)

    testData = []
    testLabel = []
    for data, label in testSet:
        testData.append([data])
        testLabel.append(label)
    testData = np.vstack(testData)
    testLabel = np.array(testLabel)

    if netType == 'CNN':
        trainData = trainData/trainData.max()
        inputShape = trainData.shape
        outputShape = tuple(list(inputShape)+[1])
        trainData = trainData.reshape(outputShape)

        testData = testData/testData.max()
        inputShape = testData.shape
        outputShape = tuple(list(inputShape)+[1])
        testData = testData.reshape(outputShape)
    elif netType == 'SNN':
        pass
    else:
        raise Exception(f'Network {netType} not available')

    return trainData, trainLabel, testData, testLabel, datasetClass
