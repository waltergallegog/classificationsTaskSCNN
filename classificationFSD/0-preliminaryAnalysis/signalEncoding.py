import sys
sys.path.append('../../')
from utils.dataStandardization import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, gammatone, freqz
from utils.dataEncoding import *
from utils.dataEncodingPerformance import *
from scipy.signal.windows import *

######################################
# ##### Sample standardization ##### #
######################################
channels = 16
sourceFolder = '../../datasets/FreeSpokenDigits/datasetRaw/'
sample = DataAudio(f'{sourceFolder}0_jackson_0.wav', channels)
padding = int((8002-len(sample.data))/2)
sample.data = np.pad(sample.data, padding, 'constant')[:8000]
time = np.linspace(0, 1, 8000)


###########################
# ##### Filter bank ##### #
###########################
##### Butterworth filter banks #####
order = 2
plt.figure()
plt.title('Butterworth filter banks')
butterFilterbank = []
for fl, fh in sample.freqPoles:
    num, den = butter(N=order, Wn=(fl, fh), btype='band', fs=sample.fs)
    butterFilterbank.append([num, den])
    freq, h = freqz(num, den, worN=20000)
    plt.plot((sample.fs*0.5/np.pi)*freq, abs(h))
plt.xlabel('f(Hz)')
plt.ylabel('Gain')
sample.decomposition(butterFilterbank)
"""
##### Gammatone filterbank #####
order = 1
plt.figure()
plt.title('Gammatone filter banks')
gammatoneFilterbank = []
for f in sample.freqCentr:
    num, den = gammatone(order=order, freq=f, ftype='fir', fs=sample.fs)
    gammatoneFilterbank.append([num, den])
    freq, h = freqz(num, den, worN=20000)
    plt.plot((sample.fs*0.5/np.pi)*freq, abs(h))
plt.xlabel('f(Hz)')
plt.ylabel('Gain')
"""

########################
# ##### Encoding ##### #
########################
label_encoding = []
spike = []
recos = []

##### Temporal Contrast #####
spike_train = TemporalContrast(sample)
setting = {'tbr_factors': 1}
spike_train.TBR(setting)
label_encoding.append('TBR')
spike.append(spike_train.TBR_spike)
recos.append(spike_train.TBR_recos)

threshold = np.mean([component.max()-component.min() for component in sample.components])/10
setting = {'sf_thresholds': threshold}
spike_train.SF(setting)
label_encoding.append('SF')
spike.append(spike_train.SF_spike)
recos.append(spike_train.SF_recos)

threshold = [np.mean(np.abs(component[1:] - component[:-1])) for component in sample.components]
setting = {'mw_window': 3, 'mw_thresholds': threshold}
spike_train.MW(setting)
label_encoding.append('MW')
spike.append(spike_train.MW_spike)
recos.append(spike_train.MW_recos)

threshold = np.mean([component.max() - component.min() for component in sample.components])/10
setting = {'pfm_threshold': threshold}
spike_train.PFM(setting)
label_encoding.append('PFM')
spike.append(spike_train.PFM_spike)
recos.append(spike_train.PFM_recos)


##### Filter Optimizer #####
spike_train = FilterOptimizer(sample)
filters = boxcar(3)
setting = {'hsa_filter': filters}
spike_train.HSA(setting)
label_encoding.append('HSA')
spike.append(spike_train.HSA_spike)
recos.append(spike_train.HSA_recos)

filters = boxcar(3)
setting = {'mhsa_filter': filters, 'mhsa_threshold': 0.85}
spike_train.MHSA(setting)
label_encoding.append('MHSA')
spike.append(spike_train.MHSA_spike)
recos.append(spike_train.MHSA_recos)

filters = boxcar(3)
setting = {'bsa_filter': filters, 'bsa_threshold': 1}
spike_train.BSA(setting)
label_encoding.append('BSA')
spike.append(spike_train.BSA_spike)
recos.append(spike_train.BSA_recos)

##### Global Referenced #####
spike_train = GlobalReferenced(sample)
setting = {'phase_bit': 6}
spike_train.PHASE(setting)
label_encoding.append('PHASE')
spike.append(spike_train.PHASE_spike)
recos.append(spike_train.PHASE_recos)

setting = {'ttfs_interval': 10}
spike_train.TTFS(setting)
label_encoding.append('TTFS')
spike.append(spike_train.TTFS_spike)
recos.append(spike_train.TTFS_recos)

setting = {'N_max': 5, 't_min': 0, 't_max': 4, 'burst_length': 13}
spike_train.BURST(setting)
label_encoding.append('BURST')
spike.append(spike_train.BURST_spike)
recos.append(spike_train.BURST_recos)


########################
# ##### Plotting ##### #
########################
##### Encoding example #####
index_encoding = len(label_encoding)
spike = [list(reversed(i)) for i in spike]
for index in range(index_encoding):
    plt.figure()
    plt.title(label_encoding[index])
    plt.eventplot(np.absolute(spike[index])*time, linelengths=0.9)
    plt.yticks([])
    plt.ylabel('Channel')
    plt.xlabel('Time')

##### Root mean square error #####
plt.figure()
plt.subplot(2, 1, 1)
plt.title('RMSE')
for index in range(index_encoding):
    channel_error = rmse(sample.components, recos[index])
    plt.plot(channel_error, '-o')
plt.xlabel('Channel')
plt.ylabel('RMSE')
plt.legend(label_encoding)

##### Spike efficiency #####
plt.subplot(2, 1, 2)
plt.title('Efficiency')
for index in range(index_encoding):
    efficiency = spike_efficiency(spike[index])
    plt.plot(efficiency, '-o')
plt.xlabel('Channel')
plt.ylabel('Efficiency')
plt.legend(label_encoding)

plt.show()
