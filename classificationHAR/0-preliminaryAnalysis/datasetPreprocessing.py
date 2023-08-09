import numpy as np
import pickle

activities = [
    (0, 'A', 'walking'),
    (1, 'B', 'jogging'),
    (2, 'C', 'stairs'),
    (3, 'D', 'sitting'),
    (4, 'E', 'standing'),
    (5, 'F', 'typing'),
    (6, 'G', 'brushing_teeth'),
    (7, 'H', 'eating_soup'),
    (8, 'I', 'eating_chips'),
    (9, 'J', 'eating_pasta'),
    (10, 'K', 'drinking'),
    (11, 'L', 'eating_sandwich'),
    (12, 'M', 'kicking_soccer'),
    (13, 'O', 'catch_tennis'),
    (14, 'P', 'dribbling_basketball'),
    (15, 'Q', 'writing'),
    (16, 'R', 'clapping'),
    (17, 'S', 'folding_clothes'),
]
activityDict = {activity[1]: activity[0] for activity in activities}

datasetDevices = ['phone', 'watch']
datasets = []
for datasetDevice in datasetDevices:
    print(f'Device: {datasetDevice}')
    datasets.append({activity[1]: [] for activity in activities})
    for subject in range(51):
        print(f'\tSubject: {subject+1}/51')
        sourceFolder = f'../../datasets/HumanActivityRecognition/datasetRaw/sourceRaw/'
        file = open(f'{sourceFolder}{datasetDevice}/accel/data_16{subject:02d}_accel_{datasetDevice}.txt', 'r')
        dataAccel = [line.rstrip().replace(';', '').split(',') for line in file.readlines()]
        file.close()
        file = open(f'{sourceFolder}{datasetDevice}/gyro/data_16{subject:02d}_gyro_{datasetDevice}.txt', 'r')
        dataGyro = [line.rstrip().replace(';', '').split(',') for line in file.readlines()]
        file.close()

        dataAccelRaw = []
        for data in dataAccel:
            idSubject, activityLabel, timestamp, x, y, z = data
            dataAccelRaw.append([int(idSubject), activityLabel, int(timestamp), float(x), float(y), float(z)])
        dataGyroRaw = []
        for data in dataGyro:
            idSubject, activityLabel, timestamp, x, y, z = data
            dataGyroRaw.append([int(idSubject), activityLabel, int(timestamp), float(x), float(y), float(z)])

        for _, activityLabel, _ in activities:
            datasets[-1][activityLabel].append({'accel': [[], [], []], 'gyro': [[], [], []]})

        for data in dataAccelRaw:
            idSubject, activityLabel, timestamp, x, y, z = data
            datasets[-1][activityLabel][-1]['accel'][0].append(x)
            datasets[-1][activityLabel][-1]['accel'][1].append(y)
            datasets[-1][activityLabel][-1]['accel'][2].append(z)
        for data in dataGyroRaw:
            idSubject, activityLabel, timestamp, x, y, z = data
            datasets[-1][activityLabel][-1]['gyro'][0].append(x)
            datasets[-1][activityLabel][-1]['gyro'][1].append(y)
            datasets[-1][activityLabel][-1]['gyro'][2].append(z)

        for _, activityLabel, _ in activities:
            if len(datasets[-1][activityLabel][-1]['accel'][0]) == 0:
                del datasets[-1][activityLabel][-1]

    # minLenght =
    # for key in dataset.keys():
    #     for i in range(len(dataset[key])):
    #         if dataset[key][i]['accel'][0] == []:
    #             print()
    #

minLength = np.min([len(dataset[key]) for dataset in datasets for key in dataset.keys()])
sourceFolder = f'../../datasets/HumanActivityRecognition/datasetRaw/'
for i, device in enumerate(datasetDevices):
    for key in datasets[i].keys():
        datasets[i][key] = datasets[i][key][:minLength]
    file = open(f'{sourceFolder}dataset{device.capitalize()}.bin', 'wb')
    pickle.dump(datasets[i], file)
    file.close()
