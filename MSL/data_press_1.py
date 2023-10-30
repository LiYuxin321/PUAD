import ast
import os
import csv
import numpy as np
from sklearn import preprocessing
import torch
import time


def run():
    dataset_folder = './'

    with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        res = [row for row in csv_reader][1:]
    res = sorted(res, key=lambda k: k[0])
    label_folder = dataset_folder + dataset + '_data'
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    label_folder_train = label_folder + '/train'
    if not os.path.exists(label_folder_train):
        os.makedirs(label_folder_train)
    label_folder_test = label_folder + '/test'
    if not os.path.exists(label_folder_test):
        os.makedirs(label_folder_test)

    data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
    labels = []
    for row in data_info:
        anomalies = ast.literal_eval(row[2])
        length = int(row[-1])
        label = np.zeros([length], dtype=np.bool)
        for anomaly in anomalies:
            label[anomaly[0]:anomaly[1] + 1] = True
        labels.append(label)
    labels = np.asarray(labels)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data_train = []
    data_test = []
    for row in data_info:
        filename = row[0]
        temp = np.load(os.path.join(dataset_folder, 'test', filename + '.npy'))
        temp = min_max_scaler.fit_transform(temp)
        data_test.append(temp.T)
        temp = np.load(os.path.join(dataset_folder, 'train', filename + '.npy'))
        temp = min_max_scaler.fit_transform(temp)
        data_train.append(temp.T)

    # processing train data
    print('processing train data ...')
    for train_i in range(len(data_train)):
        label_folder_train_temp = label_folder_train + '/' + data_info[train_i][0]
        if not os.path.exists(label_folder_train_temp):
            os.makedirs(label_folder_train_temp)

        scaled_data = data_train[train_i]
        rectangle_samples = []

        for j in range(l):
            rectangle_sample = []

            for i in range(0, scaled_data.shape[1] - win_size, l):
                if i + j <= scaled_data.shape[1] - win_size:
                    scaled_data_tmp = scaled_data[:, i + j:i + j + win_size]
                    rectangle_sample.append(scaled_data_tmp.tolist())

            rectangle_samples.append(np.array(rectangle_sample))
        sample_id = 1
        for i in range(len(rectangle_samples)):
            for data_id in range(T, len(rectangle_samples[i])):
                kpi_data = rectangle_samples[i][data_id - T:data_id]
                kpi_data = torch.tensor(kpi_data).unsqueeze(1)
                data = {'ts': np.array(0),
                        'label': np.array(0),
                        'value': kpi_data}

                path_temp = os.path.join(label_folder_train_temp, str(sample_id))
                torch.save(data, path_temp + '.seq')
                sample_id += 1

    # processing test data
    print('processing test data ...')
    for test_i in range(len(data_test)):
        label_folder_test_temp = label_folder_test + '/' + data_info[test_i][0]
        if not os.path.exists(label_folder_test_temp):
            os.makedirs(label_folder_test_temp)

        scaled_data = data_test[test_i]
        raw_label = np.expand_dims(labels[test_i], axis=0)
        rectangle_samples = []
        rectangle_labels = []

        for j in range(l):
            rectangle_sample = []
            rectangle_label = []
            for i in range(0, scaled_data.shape[1] - win_size, l):
                if i + j <= scaled_data.shape[1] - win_size:
                    scaled_data_tmp = scaled_data[:, i + j:i + j + win_size]
                    rectangle_sample.append(scaled_data_tmp.tolist())
                    raw_label_tmp = raw_label[:, i + j:i + j + win_size]
                    rectangle_label.append(raw_label_tmp.tolist())
            rectangle_samples.append(np.array(rectangle_sample))
            rectangle_labels.append(np.array(rectangle_label))
        sample_id = 1
        for i in range(len(rectangle_samples)):
            for data_id in range(T, len(rectangle_samples[i])):
                kpi_data = rectangle_samples[i][data_id - T:data_id]
                kpi_label = rectangle_labels[i][data_id - T:data_id]
                kpi_data = torch.tensor(kpi_data).unsqueeze(1)
                data = {'ts': np.array(0),
                        'label': kpi_label,
                        'value': kpi_data}

                path_temp = os.path.join(label_folder_test_temp, str(sample_id))
                torch.save(data, path_temp + '.seq')
                sample_id += 1

        a = 0


if __name__ == '__main__':
    dataset = 'MSL'
    T = 20
    win_size = 1
    l = 1

    run()
