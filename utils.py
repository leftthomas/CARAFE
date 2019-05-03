import os

import dcase_util
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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


def process_data(data_set, data_type):
    # prepare feature extractor
    extractor = dcase_util.features.MelExtractor(fs=48000, fmax=24000)
    # define feature storage path
    features_path = 'data/{}/{}'.format(data_type, 'features')
    # make sure path exists
    dcase_util.utils.Path().create(features_path)

    # loop over all audio files in the dataset and extract features for them.
    for audio_filename in tqdm(data_set.audio_files, desc='processing data for {}'.format(data_type)):
        # get filename for feature data from audio filename
        feature_filename = os.path.join(features_path, os.path.split(audio_filename)[1].replace('.wav', '.cpickle'))
        # load audio data
        audio = dcase_util.containers.AudioContainer().load(filename=audio_filename, mono=True, fs=extractor.fs)
        # extract features and store them into FeatureContainer, and save them to the disk
        feature = dcase_util.containers.FeatureContainer(data=extractor.extract(audio.data), time_resolution=extractor
                                                         .hop_length_seconds)
        feature.save(filename=feature_filename)


def load_data(data_type, batch_size=32):
    if data_type == 'DCASE2018A':
        data_set = dcase_util.datasets.TUTUrbanAcousticScenes_2018_DevelopmentSet(storage_name='raw_data',
                                                                                  data_path='data/{}'.format(data_type))
    elif data_type == 'DCASE2018B':
        data_set = dcase_util.datasets.TUTUrbanAcousticScenes_2018_Mobile_DevelopmentSet(storage_name='raw_data',
                                                                                         data_path='data/{}'.format(
                                                                                             data_type))
    elif data_type == 'DCASE2019A':
        data_set = dcase_util.datasets.TAUUrbanAcousticScenes_2019_DevelopmentSet(storage_name='raw_data',
                                                                                  data_path='data/{}'.format(data_type))
    elif data_type == 'DCASE2019B':
        data_set = dcase_util.datasets.TAUUrbanAcousticScenes_2019_Mobile_DevelopmentSet(storage_name='raw_data',
                                                                                         data_path='data/{}'.format(
                                                                                             data_type))
    else:
        raise NotImplementedError('{} is not implemented'.format(data_type))
    data_set.initialize()
    process_data(data_set, data_type)

    train_records, test_records = train_test_split(raw_data, test_size=0.3, stratify=raw_data['genre'].values)
    val_records, test_records = train_test_split(test_records, test_size=0.3, stratify=test_records['genre'].values)
    train_set, val_set, test_set = MusicData(train_records), MusicData(val_records), MusicData(test_records)
    print('# {} dataset --- train: {:d} val: {:d} test: {:d}'.format(data_type, len(train_records), len(val_records),
                                                                     len(test_records)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
