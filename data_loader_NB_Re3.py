# training and testing data loader for network

import numpy as np
from torch.utils.data import Dataset, DataLoader


class MyTrainSet(Dataset):

    def __init__(self, npy_path, prefix):
        
        super(MyTrainSet, self).__init__()

        self.srs_data = np.load(npy_path + '/SRef_train_' + prefix + '.npy').astype(np.float32)

        self.masks_data = np.load(npy_path + '/Cloud_train_' + prefix + '.npy').astype(np.float32)

        self.lstm_intervals = np.load(npy_path + '/LSTM_intervals_train_' + prefix + '.npy').astype(np.float32)
        self.lstm_intervals_inver = np.load(npy_path + '/LSTM_intervals_train_inver_' + prefix + '.npy').astype(np.float32)

        self.ndvis_data = np.load(npy_path + '/NDVI_train_' + prefix + '.npy').astype(np.float32)

    def __len__(self):
        return self.srs_data.shape[0]

    def __getitem__(self, idx):
        
        srs_data = self.srs_data[idx]

        masks_data = self.masks_data[idx]

        lstm_intervals = self.lstm_intervals[idx]
        lstm_intervals_inver = self.lstm_intervals_inver[idx]

        ndvis_data = self.ndvis_data[idx]

        return_item = [srs_data, masks_data, lstm_intervals, lstm_intervals_inver, ndvis_data]

        return return_item


def get_train_loader(batch_size, npy_path, prefix, shuffle):
    data_set = MyTrainSet(npy_path, prefix)
    data_iter = DataLoader(dataset=data_set, batch_size=batch_size, num_workers=4, shuffle=shuffle, pin_memory=True)

    return data_iter


class MyTestSet(Dataset):

    def __init__(self, npy_path, prefix):
        
        super(MyTestSet, self).__init__()

        self.srs_data = np.load(npy_path + '/SRef_train_' + prefix + '.npy').astype(np.float32)

        self.masks_data = np.load(npy_path + '/Cloud_train_' + prefix + '.npy').astype(np.float32)

        self.lstm_intervals = np.load(npy_path + '/LSTM_intervals_train_' + prefix + '.npy').astype(np.float32)
        self.lstm_intervals_inver = np.load(npy_path + '/LSTM_intervals_train_inver_' + prefix + '.npy').astype(np.float32)

        self.ndvis_data = np.load(npy_path + '/NDVI_train_' + prefix + '.npy').astype(np.float32)

    def __len__(self):
        
        return self.srs_data.shape[0]

    def __getitem__(self, idx):
        
        srs_data = self.srs_data[idx]

        masks_data = self.masks_data[idx]

        lstm_intervals = self.lstm_intervals[idx]
        lstm_intervals_inver = self.lstm_intervals_inver[idx]

        ndvis_data = self.ndvis_data[idx]

        return_item = [srs_data, masks_data, lstm_intervals, lstm_intervals_inver, ndvis_data]

        return return_item


def get_test_loader(batch_size, npy_path, prefix, shuffle):
    
    data_set = MyTestSet(npy_path, prefix)
    data_iter = DataLoader(dataset=data_set, batch_size=batch_size, num_workers=4, shuffle=shuffle, pin_memory=False)

    return data_iter
