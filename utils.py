import os

import dcase_util
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MusicData(Dataset):
    def __init__(self, data_set):
        self.samples = records['spectrogram'].values
        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(records['genre'].unique()))}
        # convert the list of label names into an array of label indices
        self.targets = np.array([self.label2index[label] for label in records['genre'].values], dtype=int)
        # load normalizer
        self.normalizer = dcase_util.data.Normalizer().load(filename='norm_factors.cpickle')

    def __getitem__(self, index):
        sample, target = self.samples[index], self.targets[index]
        normalizer.normalize(data)
        return torch.from_numpy(sample.astype(np.float32)).unsqueeze(dim=0), torch.from_numpy(np.array(target))

    def __len__(self):
        return len(self.samples)


def process_data(data_set, data_type):
    features_path = 'data/{}/{}'.format(data_type, 'features')
    if not os.path.exists(features_path):
        # prepare feature extractor
        extractor = dcase_util.features.MelExtractor(fs=48000, fmax=24000, hop_length_seconds=0.02)
        # make sure path exists
        dcase_util.utils.Path().create(features_path)
        # loop over all audio files in the dataset and extract features for them
        for audio_filename in tqdm(data_set.train_files(), desc='processing features for {}'.format(data_type)):
            # get filename for feature data from audio filename
            feature_filename = os.path.join(features_path, os.path.split(audio_filename)[1].replace('.wav', '.cpickle'))
            # load audio data
            audio = dcase_util.containers.AudioContainer().load(filename=audio_filename, mono=True, fs=extractor.fs)
            # extract features and store them into FeatureContainer, and save them to the disk
            feature = dcase_util.containers.FeatureContainer(data=extractor.extract(audio), time_resolution=0.02)
            feature.save(filename=feature_filename)

    if not os.path.exists('data/{}/{}'.format(data_type, 'norm_factors.cpickle')):
        # initialize normalizer
        normalizer = dcase_util.data.Normalizer()
        # loop over all train files in the dataset and calculate normalizer
        for audio_filename in tqdm(data_set.train_files(fold=1), desc='generating normalizer for {}'.format(data_type)):
            feature_filename = os.path.join(features_path, os.path.split(audio_filename)[1].replace('.wav', '.cpickle'))
            # load audio features
            feature = dcase_util.containers.FeatureContainer(time_resolution=0.02).load(filename=feature_filename)
            # accumulate -- feed data per file in
            normalizer.accumulate(feature)
        # after accumulation, calculate normalization factors (mean + std)
        normalizer.finalize()
        # save normalizer
        normalizer.save(filename='data/{}/{}'.format(data_type, 'norm_factors.cpickle'))


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

    train_files, val_files = data_set.validation_split(fold=1, split_type='balanced', validation_amount=0.3)
    test_files = data_set.eval(fold=1).unique_files
    train_set, val_set, test_set = MusicData(train_files), MusicData(val_files), MusicData(test_files)
    print('# {} dataset --- train: {:d} val: {:d} test: {:d}'.format(data_type, len(train_files), len(val_files),
                                                                     len(test_files)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
