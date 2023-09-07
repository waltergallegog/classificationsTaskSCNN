from copy import deepcopy
import numpy as np


def rmse(data_row, data_prediction):
    error = []
    if len(data_row) == len(data_prediction):
        data = list(zip(data_row, data_prediction))
        for row, prediction in data:
            error.append(np.sqrt(np.square(np.subtract(row, prediction)).mean()))
    return error


def spikeEfficiency(data):
    channels = deepcopy(data)
    efficiency = []
    for channel in channels:
        efficiency.append(1 - (np.sum(np.abs(channel)) / channel.shape[0]))
    return efficiency
