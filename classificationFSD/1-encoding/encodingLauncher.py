import sys
sys.path.append('../../')
import os
from utils.graphicalUtility import *

##############################
# ##### Spike encoding ##### #
##############################
filterbanks = ['butterworth', 'gammatone']
channels = [32, 64]
for filterbank in filterbanks:
    for channel in channels:
        os.system(f'python encodingSpike.py -f={filterbank} -c={channel}')


###################################
# ##### Sonogram generation ##### #
###################################
##### Encoding algorithm selected #####
encodings = ['RATE', 'TBR', 'SF', 'MW', 'PFM', 'HSA', 'MHSA', 'BSA', 'PHASE', 'TTFS', 'BURST']

##### Binning settings #####
configurations = [
    ('butterworth', 32, 20),  # (32, 70)big bins  (32, 10)small bins
    ('gammatone', 32, 20),

    ('butterworth', 32, 31.25),
    ('gammatone', 32, 31.25),

    ('butterworth', 64, 20),
    ('gammatone', 64, 20),

    ('butterworth', 64, 15.625),
    ('gammatone', 64, 15.625),
]

for encoding in encodings:
    progressBar(0, len(configurations), f'Encoding: {encoding}')
    p = 1
    for configuration in configurations:
        filterbank, channel, binsWindow = configuration
        os.system(f'python encodingSonogram.py -e={encoding} -f={filterbank} -c={channel} -b={binsWindow}')
        progressBar(p, len(configurations), f'Encoding: {encoding}')
        p += 1
