import os

import numpy as np
import torch
from dcase_util.containers import FeatureContainer, AudioContainer
from dcase_util.data import Normalizer
from dcase_util.datasets import TAUUrbanAcousticScenes_2019_Mobile_DevelopmentSet, \
    TUTUrbanAcousticScenes_2018_Mobile_DevelopmentSet, TUTUrbanAcousticScenes_2018_DevelopmentSet, \
    TAUUrbanAcousticScenes_2019_DevelopmentSet
from dcase_util.features import MelExtractor
from dcase_util.processors import FeatureReadingProcessor, NormalizationProcessor, DataShapingProcessor, \
    SequencingProcessor
from dcase_util.utils import Path
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MusicData(Dataset):
    def __init__(self, data_list, data_type):
        self.samples = data_list['file_name'].values
        self.label2index = {label: index for index, label in enumerate(sorted(data_list['file_label'].unique()))}
        self.targets = np.array([self.label2index[label] for label in data_list['file_label'].values], dtype=int)
        self.frp = FeatureReadingProcessor()
        self.np = NormalizationProcessor(filename='data/{}/{}'.format(data_type, 'norm_factors.cpickle'))
        self.sp = SequencingProcessor(sequence_length=500, hop_length=500)
        self.dsp = DataShapingProcessor(axis_list=['sequence_axis', 'data_axis', 'time_axis'])

    def __getitem__(self, index):
        sample = self.dsp.process(self.sp.process(self.np.process(self.frp.process(filename=self.samples[index]))))
        target = self.targets[index]
        return torch.from_numpy(sample.data.astype(np.float32)), torch.from_numpy(np.array(target))

    def __len__(self):
        return len(self.samples)


def process_data(data_set, data_type):
    features_path = 'data/{}/{}'.format(data_type, 'features')
    if not os.path.exists(features_path):
        # prepare feature extractor
        extractor = MelExtractor(fs=48000, fmax=24000, hop_length_seconds=0.02)
        # make sure path exists
        Path().create(features_path)
        # loop over all audio files in the dataset and extract features for them
        for audio_filename in tqdm(data_set.train_files(), desc='processing features for {}'.format(data_type)):
            # get filename for feature data from audio filename
            feature_filename = os.path.join(features_path, os.path.split(audio_filename)[1].replace('.wav', '.cpickle'))
            # load audio data
            audio = AudioContainer().load(filename=audio_filename, mono=True, fs=extractor.fs)
            # extract features and store them into FeatureContainer, and save them to the disk
            feature = FeatureContainer(data=extractor.extract(audio), time_resolution=0.02)
            feature.save(filename=feature_filename)

    if not os.path.exists('data/{}/{}'.format(data_type, 'norm_factors.cpickle')):
        # initialize normalizer
        normalizer = Normalizer()
        # loop over all train files in the dataset and calculate normalizer
        for audio_filename in tqdm(data_set.train_files(fold=1), desc='generating normalizer for {}'.format(data_type)):
            feature_filename = os.path.join(features_path, os.path.split(audio_filename)[1].replace('.wav', '.cpickle'))
            # load audio features
            feature = FeatureContainer(time_resolution=0.02).load(filename=feature_filename)
            # accumulate -- feed data per file in
            normalizer.accumulate(feature)
        # after accumulation, calculate normalization factors (mean + std)
        normalizer.finalize()
        # save normalizer
        normalizer.save(filename='data/{}/{}'.format(data_type, 'norm_factors.cpickle'))


def load_data(data_type, batch_size=32):
    if data_type == 'DCASE2018A':
        data_set = TUTUrbanAcousticScenes_2018_DevelopmentSet(storage_name='raw_data',
                                                              data_path='data/{}'.format(data_type))
    elif data_type == 'DCASE2018B':
        data_set = TUTUrbanAcousticScenes_2018_Mobile_DevelopmentSet(storage_name='raw_data',
                                                                     data_path='data/{}'.format(data_type))
    elif data_type == 'DCASE2019A':
        data_set = TAUUrbanAcousticScenes_2019_DevelopmentSet(storage_name='raw_data',
                                                              data_path='data/{}'.format(data_type))
    elif data_type == 'DCASE2019B':
        data_set = TAUUrbanAcousticScenes_2019_Mobile_DevelopmentSet(storage_name='raw_data',
                                                                     data_path='data/{}'.format(data_type))
    else:
        raise NotImplementedError('{} is not implemented'.format(data_type))
    data_set.initialize()
    process_data(data_set, data_type)

    train_files, val_files = data_set.validation_split(fold=1, split_type='balanced', validation_amount=0.3)
    test_files = data_set.eval(fold=1).unique_files

    filtered_train, filtered_val, filtered_test = [], [], []
    print('loading data, it will take a while')
    for audio_filename in train_files:
        file_name = os.path.join('data/{}/{}'.format(data_type, 'features'),
                                 os.path.split(audio_filename)[1].replace('.wav', '.cpickle'))
        file_label = data_set.file_meta(filename=audio_filename).unique_scene_labels[0]
        filtered_train.append({'file_name': file_name, 'file_label': file_label})
    for audio_filename in val_files:
        file_name = os.path.join('data/{}/{}'.format(data_type, 'features'),
                                 os.path.split(audio_filename)[1].replace('.wav', '.cpickle'))
        file_label = data_set.file_meta(filename=audio_filename).unique_scene_labels[0]
        filtered_val.append({'file_name': file_name, 'file_label': file_label})
    for audio_filename in test_files:
        file_name = os.path.join('data/{}/{}'.format(data_type, 'features'),
                                 os.path.split(audio_filename)[1].replace('.wav', '.cpickle'))
        file_label = data_set.file_meta(filename=audio_filename).unique_scene_labels[0]
        filtered_test.append({'file_name': file_name, 'file_label': file_label})
    filtered_train = DataFrame.from_records(filtered_train)
    filtered_val = DataFrame.from_records(filtered_val)
    filtered_test = DataFrame.from_records(filtered_test)

    train_set, val_set, test_set = MusicData(filtered_train, data_type), MusicData(filtered_val, data_type), MusicData(
        filtered_test, data_type)
    print('# {} dataset --- train: {:d} val: {:d} test: {:d}'.format(data_type, len(train_files), len(val_files),
                                                                     len(test_files)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
