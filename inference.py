import argparse
from collections import Counter

import numpy as np
import torch
from librosa.core import load
from librosa.feature import melspectrogram

from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Music Genre Classification')
    parser.add_argument('--data_type', default='GTZAN', type=str, choices=['GTZAN', 'EBallroom'], help='dataset type')
    parser.add_argument('--music_name', type=str, help='test music name')
    parser.add_argument('--model_name', default='GTZAN.pth', type=str, help='model epoch name')
    opt = parser.parse_args()

    DATA_TYPE = opt.data_type
    MUSIC_NAME = opt.music_name
    MODEL_NAME = opt.model_name

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if DATA_TYPE == 'GTZAN':
        model = Model(10)
    else:
        model = Model(13)
    checkpoint = torch.load('epochs/{}'.format(MODEL_NAME), map_location=lambda storage, loc: storage)
    model = model.load_state_dict(checkpoint).to(DEVICE).eval()

    y, sr = load(MUSIC_NAME, mono=True)
    S = melspectrogram(y, sr).T
    S = S[:-1 * (S.shape[0] % 128)]
    num_chunk = S.shape[0] / 128
    data_chunks = np.split(S, num_chunk)
    genres = list()
    for i, data in enumerate(data_chunks):
        data = torch.FloatTensor(data).view(1, 1, 128, 128)
        preds = model(data)
        pred_val, pred_index = preds.max(1)
        pred_index = pred_index.cpu().numpy()[0]
        pred_val = np.exp(pred_val.cpu().numpy()[0])
        pred_genre = le.inverse_transform(pred_index)
        if pred_val >= 0.5:
            genres.append(pred_genre)
    # ------------------------------- #
    s = float(sum([v for k, v in dict(Counter(genres)).items()]))
    pos_genre = sorted([(k, v / s * 100) for k, v in dict(Counter(genres)).items()], key=lambda x: x[1], reverse=True)
    for genre, pos in pos_genre:
        print("%10s: \t%.2f\t%%" % (genre, pos))
