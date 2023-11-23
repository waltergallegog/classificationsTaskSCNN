from copy import deepcopy
import random
import numpy as np
import math

"""
L'utilizzo del temporal coding e giustificato dal fatto che il rate coding ha un processo di trasmissione dati molto inefficiente
vedere paper 1998 Rate coding versus temporal order coding: a theoretical approach
"""


class AddressEventRepresentation:
    def __init__(self, spike_train):
        spike_train = np.vstack(spike_train)
        self.addresses = []
        self.timestamps = []
        for x in range(spike_train.shape[1]):
            for y in range(spike_train.shape[0]):
                if spike_train[y, x] != 0:
                    self.addresses.append(y)
                    self.timestamps.append(x)
        self.addresses = np.array(self.addresses)
        self.timestamps = np.array(self.timestamps)


class DataInitializing:
    def __init__(self, data, fs=False):
        if not fs:
            self.data = data.components
            self.fs = data.fs
        elif fs:
            self.data = data
            self.fs = fs
        self.refra_t = 0*int(self.fs * 0.003)


class RateCoding(DataInitializing):
    def RATE(self, setting):  # Poisson rate
        self.RATE_spike = []
        self.RATE_recos = []
        data = deepcopy(self.data)
        random.seed(0)
        TIME_STIMULUS = 1
        channels = range(len(data))
        for channel in channels:
            signal = data[channel]
            tmp = []

            # we are generating an array of spike times for each channel
            # Sparse representation
            for i in range(len(signal)):
                rate = signal[i]
                if rate == 0 or rate < 0:
                    tmp.append(np.array([]))
                else:
                    spike_sequence = []
                    poisson_isi = -math.log(1.0 - random.random()) / rate * 1000.0  # ms tau
                    spike_time = poisson_isi
                    while spike_time < TIME_STIMULUS:
                        spike_sequence.append(spike_time)
                        poisson_isi = -math.log(1.0 - random.random()) / rate * 1000.0  # ms tau
                        spike_time += poisson_isi
                    tmp.append(np.array(spike_sequence))

            # convert them from spike times to spike trains
            # explicit representation
            spike_time = []
            for i in range(len(tmp)):
                scaled = np.array(tmp[i]) + TIME_STIMULUS * i
                for t in scaled:
                    spike_time.append(int(t))
            tmp = np.zeros(int(TIME_STIMULUS) * len(signal))
            for i in spike_time:
                tmp[i] = 1
            self.RATE_spike.append(tmp)
        self.RATE_aer = AddressEventRepresentation(self.RATE_spike)


class TemporalContrast(DataInitializing):
    def TBR(self, setting):  # Threshold-based representation
        """
        Il TBR e' un algoritmo che permette di filtrare il rumore attraverso il calcolo della media e della
        deviazione standatd, quanto piu' il segnale restera' in uno stato rumoroso o nello range, piu' il threshold si
        terra intorno a valori che il segnale mantiene di piu, il parametro factor ci permette di definire
        l'importanza dell'informazione contenuta in quel livello se
        - facotor e grande vengono considerati solo segnali al di fuori del rumore
        - factor e piccolo tutta l'informazione viene codificata
        per segnali privi di rumore cio' che e' codificato sono le rapide variazioni di segnale tanto piu' grande e' il
        factor

        :param setting:
        :return:
        """
        self.TBR_spike = []
        self.TBR_recos = []
        data = deepcopy(self.data)

        factor = setting['tbr_factors']

        channels = range(len(data))
        for channel in channels:
            t_spike = np.infty
            variation = data[channel][1:] - data[channel][:-1]
            threshold = np.mean(variation) + factor * np.std(variation)
            variation = np.insert(variation, 0, variation[1])
            length_array = data[channel].shape[0]
            base = data[channel][0]
            tmp_spike = np.zeros(length_array, dtype=int)
            tmp_recos = np.zeros(length_array)
            for index in range(length_array):
                if variation[index] > threshold and t_spike > self.refra_t:
                    tmp_spike[index] = 1
                    base += threshold
                    t_spike = 0
                elif variation[index] < -threshold and t_spike > self.refra_t:
                    tmp_spike[index] = -1
                    base -= threshold
                    t_spike = 0
                else:
                    tmp_spike[index] = 0
                    t_spike += 1
                tmp_recos[index] = base
            self.TBR_spike.append(tmp_spike)
            self.TBR_recos.append(np.array(tmp_recos))

        self.TBR_aer = AddressEventRepresentation(self.TBR_spike)

    def SF(self, setting):  # Step-forward
        """
        Nello step-forward il thresholds viene imposto in base alla risoluzione del segnale che vogliamo modellare
        buon criterio 1/10 della massima escursione media

        :param setting:
        :return:
        """
        self.SF_spike = []
        self.SF_recos = []
        data = deepcopy(self.data)

        THRESHOLD = setting['sf_thresholds']

        t_spike = np.infty
        channels = range(len(data))
        for channel in channels:
            length_array = data[channel].shape[0]
            base = data[channel][0]
            tmp_spike = np.zeros(length_array, dtype=int)
            tmp_recos = np.zeros(length_array)
            for index in range(length_array):
                if data[channel][index] > base + THRESHOLD and t_spike > self.refra_t:
                    tmp_spike[index] = 1
                    base += THRESHOLD
                    t_spike = 0
                elif data[channel][index] < base - THRESHOLD and t_spike > self.refra_t:
                    tmp_spike[index] = -1
                    base -= THRESHOLD
                    t_spike = 0
                else:
                    tmp_spike[index] = 0
                    t_spike += 1
                tmp_recos[index] = base
            self.SF_spike.append(tmp_spike)
            self.SF_recos.append(np.array(tmp_recos))

        self.SF_aer = AddressEventRepresentation(self.SF_spike)

    def MW(self, setting):  # moving window
        """
        In questo caso la finestra e il threshold sono collegati tra loro, il criterio di scelta per segnali sinusoidali
        per la scelta dei parametri e dato da:
        threshold = media sulle variazioni su tutto il segnale tra due istanti successivi
        threshold = [np.mean(np.abs(component[1:] - component[:-1])) for component in sample.components]

        window = usando il criterio precedente la finestra ottimale e tra 2 e 4 permettendo di eliminare segnali poco
        variabili
        :param setting:
        :return:
        """
        self.MW_spike = []
        self.MW_recos = []
        data = deepcopy(self.data)

        threshold = setting['mw_thresholds']
        window = setting['mw_window']

        channels = range(len(data))
        for channel in channels:
            length_array = data[channel].shape[0]
            tmp_spike = np.zeros(length_array, dtype=int)
            tmp_recos = np.zeros(length_array)
            base = np.mean(data[channel][0:window])
            base_recos = data[channel][0]
            for index in range(window):
                if data[channel][index] > base + threshold[channel]:
                    tmp_spike[index] = 1
                    base_recos += threshold[channel]
                elif data[channel][index] < base - threshold[channel]:
                    tmp_spike[index] = -1
                    base_recos -= threshold[channel]
                else:
                    tmp_spike[index] = 0
                tmp_recos[index] = base_recos

            t_spike = np.infty
            for index in range(window, length_array):
                base = np.mean(data[channel][(index - window):index])
                if data[channel][index] > base + threshold[channel] and t_spike > self.refra_t:
                    tmp_spike[index] = 1
                    t_spike = 0
                    base_recos += threshold[channel]
                elif data[channel][index] < base - threshold[channel] and t_spike > self.refra_t:
                    tmp_spike[index] = -1
                    t_spike = 0
                    base_recos -= threshold[channel]
                else:
                    tmp_spike[index] = 0
                    t_spike += 1
                tmp_recos[index] = base_recos
            self.MW_spike.append(tmp_spike)
            self.MW_recos.append(np.array(tmp_recos))

        self.MW_aer = AddressEventRepresentation(self.MW_spike)

    def PFM(self, setting):
        self.PFM_spike = []
        self.PFM_recos = []
        data = deepcopy(self.data)
        data = [np.where(signal < 0, 0, signal) for signal in data]

        threshold = setting['pfm_threshold']

        t_spike = np.infty
        channels = range(len(data))
        for channel in channels:
            length_array = data[channel].shape[0]
            tmp_spike = np.zeros(length_array, dtype=int)
            tmp_recos = np.zeros(length_array)
            for index in range(length_array):
                if data[channel][index] > threshold and t_spike > self.refra_t:
                    tmp_spike[index] = 1
                    t_spike = 0
                    tmp_recos[index] = threshold
                else:
                    tmp_spike[index] = 0
                    t_spike += 1
                    tmp_recos[index] = 0
            self.PFM_spike.append(tmp_spike)
            self.PFM_recos.append(tmp_recos)
        self.PFM_aer = AddressEventRepresentation(self.PFM_spike)


class FilterOptimizer(DataInitializing):
    def HSA(self, setting):
        """
        Il filtro deve essere scelto in base al segnale in questo caso agni filtro e' in grado di modellare correttamente
        il segnale, aspetto importante e' la finestra che in questo caso e' fissata a 3 valore minimo che permette di
        codificare in modo corretto il segnale
        from scipy.signal.windows import *
        filters = boxcar(3)
        :param setting:
        :return:
        """
        self.HSA_spike = []
        self.HSA_recos = []
        data = deepcopy(self.data)

        filters = setting['hsa_filter']

        t_spike = np.infty

        length_filter = len(filters)
        channels = range(len(data))
        for channel in channels:
            length_array = data[channel].shape[0]
            tmp = np.zeros(length_array, dtype=int)
            for index in range(length_array):
                counter = 0
                t_spike += 1
                for cell in range(length_filter):
                    if index + cell < length_array and data[channel][index + cell] >= filters[cell]:
                        counter = counter + 1
                    else:
                        break
                if counter == length_filter:
                    for cell in range(length_filter):
                        if index + cell < length_array:
                            data[channel][index + cell] = data[channel][index + cell] - filters[cell]
                    if t_spike > self.refra_t:
                        t_spike = 0
                        tmp[index] = 1
            recos = np.convolve(tmp, filters)
            recos = recos[0:len(data[channel])]
            self.HSA_spike.append(tmp)
            self.HSA_recos.append(np.array(recos))

        self.HSA_aer = AddressEventRepresentation(self.HSA_spike)

    def MHSA(self, setting):
        """
        from scipy.signal.windows import *
        fil = triang(3)
        setting = {'mhsa_filter': fil, 'mhsa_threshold': 0.85}
        :param setting:
        :return:
        """
        self.MHSA_spike = []
        self.MHSA_recos = []
        data = deepcopy(self.data)

        filters = setting['mhsa_filter']
        threshold = setting['mhsa_threshold']

        t_spike = np.infty

        length_filter = len(filters)
        channels = range(len(data))
        for channel in channels:
            length_array = data[channel].shape[0]
            tmp = np.zeros(length_array, dtype=int)
            for index in range(length_array):
                error = 0
                t_spike += 1
                for cell in range(length_filter):
                    if index + cell < length_array and data[channel][index + cell] < filters[cell]:
                        error = error + filters[cell] - data[channel][index + cell]
                if error <= threshold:
                    for cell in range(length_filter):
                        if index + cell < length_array:
                            data[channel][index + cell] = data[channel][index + cell] - filters[cell]
                    if t_spike > self.refra_t:
                        t_spike = 0
                        tmp[index] = 1
            recos = np.convolve(tmp, filters)
            recos = recos[0:len(data[channel])]
            self.MHSA_spike.append(tmp)
            self.MHSA_recos.append(np.array(recos))

        self.MHSA_aer = AddressEventRepresentation(self.MHSA_spike)

    def BSA(self, setting):
        """
        from scipy.signal.windows import *
        fil = triang(3)
        setting = {'bsa_filter': fil, 'bsa_threshold': 1}
        :param setting:
        :return:
        """
        self.BSA_spike = []
        self.BSA_recos = []
        data = deepcopy(self.data)

        filters = setting['bsa_filter']
        threshold = setting['bsa_threshold']

        t_spike = np.infty

        length_filter = len(filters)
        channels = range(len(data))

        for channel in channels:
            length_array = data[channel].shape[0]
            tmp = np.zeros(length_array, dtype=int)
            for index in range(length_array - length_filter + 1):
                error1 = 0
                error2 = 0
                t_spike += 1
                for cell in range(length_filter):
                    error1 = error1 + abs(data[channel][index + cell] - filters[cell])
                    error2 = error2 + abs(data[channel][index + cell])
                if error1 <= error2 * threshold:
                    for cell in range(length_filter):
                        if index + cell + 1 < length_array:
                            data[channel][index + cell] = data[channel][index + cell] - filters[cell]
                    if t_spike > self.refra_t:
                        t_spike = 0
                        tmp[index] = 1
            recos = np.convolve(tmp, filters)
            recos = recos[0:len(data[channel])]
            self.BSA_spike.append(tmp)
            self.BSA_recos.append(np.array(recos))

        self.BSA_aer = AddressEventRepresentation(self.BSA_spike)


class GlobalReferenced(DataInitializing):
    def PHASE(self, setting):
        self.PHASE_spike = []
        self.PHASE_recos = []
        data = deepcopy(self.data)

        bit = setting['phase_bit']

        tmp = [np.where(signal < 0, 0, signal) for signal in data]
        data_redux = []
        for signal in tmp:
            data_redux.append(np.array([np.mean(signal[i:i + bit]) for i in range(0, len(signal), bit)]))
        normalization = np.max([signal.max() for signal in data_redux])
        data_redux = [np.arcsin(signal / normalization) for signal in data_redux]
        level = lambda x, bit: np.pi / (2 ** (bit + 1)) * x
        channels = range(len(data_redux))
        for channel in channels:
            length_array = data_redux[channel].shape[0]
            tmp_spike = []
            tmp_recos = []
            for index in range(length_array):
                point = data_redux[channel][index]
                for i in range(2 ** bit):
                    if level(i, bit) <= point < level(i + 1, bit):
                        code = format(i, '0{}b'.format(bit))
                        code = reversed(code)
                        [tmp_spike.append(int(q)) for q in code]
                        level_recos = (level(i, bit) + level(i + 1, bit)) / 2
                        [tmp_recos.append(level_recos) for q in range(bit)]
                        break
                if point >= level(2 ** bit - 1, bit):
                    code = format(2 ** bit - 1, '0{}b'.format(bit))
                    code = reversed(code)
                    [tmp_spike.append(int(q)) for q in code]
                    level_recos = level(2 ** bit - 1, bit)
                    [tmp_recos.append(level_recos) for q in range(bit)]
            tmp_spike = np.array(tmp_spike)
            len_flag = data[channel].shape[0]
            if tmp_spike.shape[0] > len_flag:
                tmp_spike = tmp_spike[0:len_flag]
            elif tmp_spike.shape[0] < len_flag:
                tmp_spike = np.concatenate((tmp_spike, np.zeros(len_flag - tmp_spike.shape[0])))
            for index in range(len_flag):
                if tmp_spike[index] == 1:
                    tmp_spike[index + 1:index + self.refra_t] = 0

            tmp_recos = np.sin(np.array(tmp_recos)) * normalization
            if tmp_recos.shape[0] > len_flag:
                tmp_recos = tmp_recos[0:len_flag]
            elif tmp_recos.shape[0] < len_flag:
                tmp_recos = np.concatenate((tmp_recos, np.zeros(len_flag - tmp_recos.shape[0])))
            for index in range(len_flag):
                if tmp_recos[index] == 1:
                    tmp_recos[index + 1:index + self.refra_t] = 0
            self.PHASE_spike.append(tmp_spike)
            self.PHASE_recos.append(tmp_recos)

        self.PHASE_aer = AddressEventRepresentation(self.PHASE_spike)

    def TTFS(self, setting):
        self.TTFS_spike = []
        self.TTFS_recos = []
        data = deepcopy(self.data)
        interval = setting['ttfs_interval']

        import warnings
        warnings.filterwarnings('ignore', message='.*divide by zero encountered.*')
        tmp = [np.where(signal > 0, signal, 0) for signal in data]

        data_redux = []
        for signal in tmp:
            data_redux.append(np.array([np.mean(signal[i:i + interval]) for i in range(0, len(signal), interval)]))
        normalization = np.max([signal.max() for signal in data_redux])
        data_redux = [0.1 * np.log(normalization / signal) for signal in data_redux]

        level = lambda x, interval: 1 / interval * x
        channels = range(len(data_redux))
        for channel in channels:
            length_array = data_redux[channel].shape[0]
            tmp_spike = []
            tmp_recos = []
            for index in range(length_array):
                point = data_redux[channel][index]
                code = []
                for i in range(interval):
                    if level(i, interval) <= point <= level(i + 1, interval):
                        code.append(1)
                        level_recos = (level(i, interval) + level(i + 1, interval)) / 2
                        [tmp_recos.append(level_recos) for q in range(interval)]
                    else:
                        code.append(0)
                [tmp_spike.append(int(q)) for q in code]

            tmp_spike = np.array(tmp_spike)
            len_flag = data[channel].shape[0]
            if tmp_spike.shape[0] > len_flag:
                tmp_spike = tmp_spike[0:len_flag]
            elif tmp_spike.shape[0] < len_flag:
                tmp_spike = np.concatenate((tmp_spike, np.zeros(len_flag - tmp_spike.shape[0])))
            for index in range(len_flag):
                if tmp_spike[index] == 1:
                    tmp_spike[index + 1:index + self.refra_t] = 0

            tmp_recos = np.exp(-np.array(tmp_recos) / 0.1) * normalization
            if tmp_recos.shape[0] > len_flag:
                tmp_recos = tmp_recos[0:len_flag]
            elif tmp_recos.shape[0] < len_flag:
                tmp_recos = np.concatenate((tmp_recos, np.zeros(len_flag - tmp_recos.shape[0])))
            for index in range(len_flag):
                if tmp_recos[index] == 1:
                    tmp_recos[index + 1:index + self.refra_t] = 0
            self.TTFS_spike.append(tmp_spike)
            self.TTFS_recos.append(tmp_recos)

        self.TTFS_aer = AddressEventRepresentation(self.TTFS_spike)

    def BURST(self, setting):
        self.BURST_spike = []
        self.BURST_recos = []
        data = deepcopy(self.data)

        N_max = setting['N_max']
        t_min = setting['t_min']
        t_max = setting['t_max']
        length = setting['burst_length']

        tmp = [np.where(signal < 0, 0, signal) for signal in data]
        data_redux = []
        for signal in tmp:
            data_redux.append(np.array([np.mean(signal[i:i + length]) for i in range(0, len(signal), length)]))
        normalization = np.max([signal.max() for signal in data_redux])
        data_redux = [signal / normalization for signal in data_redux]
        data_redux = [np.where(signal < 0.001, 0, signal) for signal in data_redux]

        channels = range(len(data_redux))
        for channel in channels:
            length_array = data_redux[channel].shape[0]
            tmp_spike = []
            tmp_recos = []
            for index in range(length_array):
                rate = data_redux[channel][index]
                spike_number = int(np.ceil(rate * N_max))
                if spike_number > 1:
                    isi = int(np.ceil(t_max - rate * (t_max - t_min)))
                else:
                    isi = t_max
                if length > spike_number * (isi + 1):
                    code = ([1] + isi * [0]) * spike_number
                else:
                    raise ValueError('Invalid stream length, the min length is {}'.format(spike_number * (isi + 1) + 1))
                if len(code) < length:
                    code = code + [0] * (length - len(code))
                [tmp_spike.append(int(q)) for q in code]
                rate_recos = spike_number / N_max
                [tmp_recos.append(rate_recos) for q in range(length)]

            tmp_spike = np.array(tmp_spike)
            len_flag = data[channel].shape[0]
            if tmp_spike.shape[0] > len_flag:
                tmp_spike = tmp_spike[0:len_flag]
            elif tmp_spike.shape[0] < len_flag:
                tmp_spike = np.concatenate((tmp_spike, np.zeros(len_flag - tmp_spike.shape[0])))
            for index in range(len_flag):
                if tmp_spike[index] == 1:
                    tmp_spike[index + 1:index + self.refra_t] = 0

            tmp_recos = np.array(tmp_recos) * normalization
            if tmp_recos.shape[0] > len_flag:
                tmp_recos = tmp_recos[0:len_flag]
            elif tmp_recos.shape[0] < len_flag:
                tmp_recos = np.concatenate((tmp_recos, np.zeros(len_flag - tmp_recos.shape[0])))
            for index in range(len_flag):
                if tmp_recos[index] == 1:
                    tmp_recos[index + 1:index + self.refra_t] = 0

            self.BURST_spike.append(tmp_spike)
            self.BURST_recos.append(tmp_recos)

        self.BURST_aer = AddressEventRepresentation(self.BURST_spike)
