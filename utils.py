import os
import pickle

import numpy as np
import torch
from librosa.core import load
from librosa.feature import melspectrogram
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class MusicData(Dataset):
    def __init__(self, records):
        self.samples = records['spectrogram'].values
        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(records['genre'].unique()))}
        # convert the list of label names into an array of label indices
        self.targets = np.array([self.label2index[label] for label in records['genre'].values], dtype=int)

    def __getitem__(self, index):
        sample, target = self.samples[index], self.targets[index]
        return torch.from_numpy(sample.astype(np.float32)).unsqueeze(dim=0), torch.from_numpy(np.array(target))

    def __len__(self):
        return len(self.samples)


def process_data(data_type):
    print('processing {} dataset, this will take a while, but it will be done only once.'.format(data_type))
    records = []
    for genre in sorted(os.listdir(os.path.join('data', data_type))):
        for track in sorted(os.listdir(os.path.join('data', data_type, genre))):
            y, sr = load(os.path.join('data', data_type, genre, track), mono=True)
            S = melspectrogram(y, sr).T
            S = S[:-1 * (S.shape[0] % 128)]
            num_chunk = S.shape[0] / 128
            data_chunks = np.split(S, num_chunk)
            data_chunks = [(data, genre) for data in data_chunks]
            records.append(data_chunks)

    records = DataFrame.from_records([data for record in records for data in record],
                                     columns=['spectrogram', 'genre'])
    with open('data/{}.pkl'.format(data_type), 'wb') as outfile:
        pickle.dump(records, outfile, pickle.HIGHEST_PROTOCOL)
    print('{} dataset is processed'.format(data_type))


def load_data(data_type, batch_size=32):
    if not os.path.exists('data/{}.pkl'.format(data_type)):
        process_data(data_type)
    with open('data/{}.pkl'.format(data_type), 'rb') as infile:
        raw_data = pickle.load(infile)

    train_records, test_records = train_test_split(raw_data, test_size=0.3, stratify=raw_data['genre'].values)
    val_records, test_records = train_test_split(test_records, test_size=0.3, stratify=test_records['genre'].values)
    train_set, val_set, test_set = MusicData(train_records), MusicData(val_records), MusicData(test_records)
    print('# {} dataset --- train: {:d} val: {:d} test: {:d}'.format(data_type, len(train_records), len(val_records),
                                                                     len(test_records)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
