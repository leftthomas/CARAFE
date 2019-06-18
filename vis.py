import argparse
import math

import torch
from torchvision.utils import save_image

import utils
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Capsule Network Focused Parts')
    parser.add_argument('--data_name', default='voc', type=str, choices=['voc', 'coco', 'cityscapes'],
                        help='dataset name')
    parser.add_argument('--batch_size', default=64, type=int, help='vis batch size')
    parser.add_argument('--num_iterations', default=3, type=int, help='routing iterations number')

    opt = parser.parse_args()
    DATA_NAME, BATCH_SIZE, NUM_ITERATIONS = opt.data_name, opt.batch_size, opt.num_iterations
    nrow = math.floor(math.sqrt(BATCH_SIZE))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = utils.load_data(DATA_NAME, 'test', BATCH_SIZE, shuffle=True)
    images, labels = next(iter(data_loader))
    save_image(images, filename='results/vis_{}_original.png'.format(DATA_NAME), nrow=nrow, normalize=True, padding=4,
               pad_value=255)

    model = Model(len(data_loader.dataset.classes), NUM_ITERATIONS)
    model.load_state_dict(torch.load('epochs/{}.pth'.format(DATA_NAME), map_location='cpu'))
    model, images = model.to(device), images.to(device)
    probam = utils.ProbAM(model)

    features_heat_maps = probam(images)
    save_image(features_heat_maps, filename='results/vis_{}_features.png'.format(DATA_NAME), nrow=nrow, padding=4,
               pad_value=255)
