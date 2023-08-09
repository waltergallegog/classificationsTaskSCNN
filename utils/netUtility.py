import pickle as pickleLocal
from random import shuffle as shuffleLocal
import numpy as npLocal

def datasetSplitting(fileName, netType):

    file = open(fileName, 'rb')
    sonogramSet = pickleLocal.load(file)
    file.close()

    datasetClass = sonogramSet[-1][-1]+1

    classNum = sonogramSet[-1][-1]+1
    sampleNum = int(len(sonogramSet)/classNum)
    splitTrain, splitTest = int(npLocal.floor(sampleNum*0.8)), int(npLocal.ceil(sampleNum*0.2))

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
    shuffleLocal(trainSet)
    shuffleLocal(testSet)

    trainData = []
    trainLabel = []
    for data, label in trainSet:
        trainData.append([data])
        trainLabel.append(label)
    trainData = npLocal.vstack(trainData)
    trainLabel = npLocal.array(trainLabel)

    testData = []
    testLabel = []
    for data, label in testSet:
        testData.append([data])
        testLabel.append(label)
    testData = npLocal.vstack(testData)
    testLabel = npLocal.array(testLabel)

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
