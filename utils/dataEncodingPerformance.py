from copy import deepcopy as deepcopyLocal
import numpy as npLocal


def rmse(data_row, data_prediction):
    error = []
    if len(data_row) == len(data_prediction):
        data = list(zip(data_row, data_prediction))
        for row, prediction in data:
            error.append(npLocal.sqrt(npLocal.square(npLocal.subtract(row, prediction)).mean()))
    return error


def spike_efficiency(data):
    channels = deepcopyLocal(data)
    efficiency = []
    for channel in channels:
        efficiency.append(1 - (npLocal.sum(npLocal.abs(channel)) / channel.shape[0]))
    return efficiency
