import torch
import os
import torchvision
import torch.utils.data
import torch.utils.data as data
import numpy as np
# from model2 import *
from tqdm import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from torch.utils.data import Sampler
from sklearn.preprocessing import StandardScaler
import pandas as pd


# class KpiReader(data.Dataset):
#     def __init__(self, path):
#         super(KpiReader, self).__init__()
#         self.path = path
#         self.length = len(glob.glob(self.path + '/*.seq'))
#         data = []
#         for i in range(self.length):
#             # item = torch.load(self.path+'/%d.seq' % (i+1))
#             item = self.path + '/%d.seq' % (i + 1)
#             data.append(item)
#         self.data = data
#
#     def __getitem__(self, index):
#         temp_data = torch.load(self.data[index])
#         kpi_ts, kpi_label, kpi_value = temp_data['ts'], temp_data['label'], temp_data['value']
#         return kpi_ts, kpi_label, kpi_value
#
#     def __len__(self):
#         return self.length

class KpiReader(data.Dataset):
    def __init__(self, path):
        super(KpiReader, self).__init__()
        self.path = path
        self.length = len(glob.glob(self.path + '/*.seq'))
        data = []
        for i in range(self.length):
            # item = torch.load(self.path+'/%d.seq' % (i+1))
            item = self.path + '/%d.seq' % (i + 1)
            data.append(item)
        self.data = data

    def __getitem__(self, index):
        temp_data = torch.load(self.data[index])
        kpi_ts, kpi_label, kpi_value = temp_data['ts'], temp_data['label'], temp_data['value']
        return kpi_ts, kpi_label, kpi_value

    def __len__(self):
        return self.length


class KpiReaderTrain(data.Dataset):
    def __init__(self, path):
        super(KpiReaderTrain, self).__init__()
        self.path = path
        data = []
        label = []
        for i in range(len(self.path)):
            length = len(glob.glob(self.path[i] + '/*.seq'))
            for j in range(length):
                item = self.path[i] + '/%d.seq' % (j + 1)
                data.append(item)
                label.append(i)
        self.data = data
        self.label = label
        self.length = len(data)

    def __getitem__(self, index):
        temp_data = torch.load(self.data[index])
        kpi_ts, kpi_label, kpi_value = temp_data['ts'], temp_data['label'], temp_data['value']

        # return kpi_ts, kpi_label, kpi_value, self.label[index], index
        return kpi_ts, kpi_label, kpi_value

    def __len__(self):
        return self.length


class CategoriesSampler(Sampler):
    """A Sampler to sample a FSL task.

    Args:
        Sampler (torch.utils.data.Sampler): Base sampler from PyTorch.
    """

    def __init__(
            self,
            label_list,
            label_num,
            episode_size,
            episode_num,
            way_num,
            image_num,
    ):
        """Init a CategoriesSampler and generate a label-index list.

        Args:
            label_list (list): The label list from label list.
            label_num (int): The number of unique labels.
            episode_size (int): FSL setting.
            episode_num (int): FSL setting.
            way_num (int): FSL setting.
            image_num (int): FSL setting.
        """
        super(CategoriesSampler, self).__init__(label_list)

        self.episode_size = episode_size
        self.episode_num = episode_num
        self.way_num = way_num
        self.image_num = image_num

        label_list = np.array(label_list)
        self.idx_list = []
        for label_idx in range(label_num):
            ind = np.argwhere(label_list == label_idx).reshape(-1)
            ind = torch.from_numpy(ind)
            self.idx_list.append(ind)

    def __len__(self):
        return self.episode_num

    def __iter__(self):
        """Random sample a FSL task batch(multi-task).

        Yields:
            torch.Tensor: The stacked tensor of a FSL task batch(multi-task).
        """
        batch = []
        for i_batch in range(self.episode_num):
            classes = torch.randperm(len(self.idx_list))[: self.way_num]
            for c in classes:
                idxes = self.idx_list[c.item()]
                pos = torch.randperm(idxes.size(0))[: self.image_num]
                batch.append(idxes[pos])
            if len(batch) == self.episode_size * self.way_num:
                batch = torch.stack(batch).reshape(-1)
                yield batch
                batch = []


class SMAPSegLoader(data.Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class PSMSegLoader(data.Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
