import sys
sys.path.append('../../')
from utils.dataStandardization import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, gammatone, freqz

######################################
# ##### Sample standardization ##### #
######################################
channels = 16
sourceFolder = '../../datasets/FreeSpokenDigits/datasetRaw/'
sample = DataAudio(f'{sourceFolder}0_jackson_0.wav', channels)
padding = int((8002-len(sample.data))/2)
sample.data = np.pad(sample.data, padding, 'constant')[:8000]
time = np.linspace(0, 1, 8000)

# Sample plotting
plt.figure()
plt.title('FSD Sample')
plt.plot(time, sample.data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()


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


####################################################
# ##### Frequency decomposition and Plotting ##### #
####################################################
##### Spectrogram #####
sample.decomposition(butterFilterbank)
sonogram = np.absolute(np.vstack(sample.components))
plt.figure()
plt.title('Spectrogram with Butterworth filterbank')
plt.imshow(sonogram, aspect='auto', vmax=int(sonogram.max()/10))
plt.xlabel('Time')
plt.ylabel('Channels')

##### Frequency components #####
for i in range(channels):
    if i % 16 == 0: plt.figure()
    plt.subplot(16, 1, (i % 16) + 1)
    # if i % 16 == 0: plt.title('Channel component')
    # if i == 8: plt.ylabel('Amplitude')
    plt.plot(sample.components[i]/sample.components[i].max())
    plt.axis('off')
plt.xlabel('Sample')
plt.tight_layout()
plt.show()
