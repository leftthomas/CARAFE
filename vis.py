import argparse

import torch
from torchvision.utils import save_image

from probam import ProbAM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Capsule Network and CNN Focused Parts')
    parser.add_argument('--data_type', default='STL10', type=str,
                        choices=['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'STL10'], help='dataset type')
    parser.add_argument('--data_mode', default='test_single', type=str,
                        choices=['test_single', 'test_multi'], help='visualized data mode')
    parser.add_argument('--num_iterations', default=3, type=int, help='routing iterations number')
    opt = parser.parse_args()

    DATA_TYPE = opt.data_type
    DATA_MODE = opt.data_mode
    NUM_ITERATIONS = opt.num_iterations
    batch_size = 16 if DATA_MODE == 'test_single' else 8
    nrow = 4 if DATA_MODE == 'test_single' else 2

    images, labels = next(iter(get_iterator(DATA_TYPE, DATA_MODE, batch_size, False)))
    save_image(images, filename='vis_%s_%s_original.png' % (DATA_TYPE, DATA_MODE), nrow=nrow, normalize=True, padding=4,
               pad_value=255)

    for NET_MODE in ['Capsule_ps', 'Capsule_fc', 'CNN']:
        if NET_MODE == 'Capsule_ps':
            model = MixNet(data_type=DATA_TYPE, capsule_type='ps', num_iterations=NUM_ITERATIONS, return_prob=True)
            AM_method = ProbAM(model)
        elif NET_MODE == 'Capsule_fc':
            model = MixNet(data_type=DATA_TYPE, capsule_type='fc', routing_type='dynamic',
                           num_iterations=NUM_ITERATIONS, return_prob=True)
            AM_method = ProbAM(model)
        else:
            model = MixNet(data_type=DATA_TYPE, net_mode='CNN')
            AM_method = GradCam(model)
        if torch.cuda.is_available():
            model = model.to('cuda')
            model.load_state_dict(torch.load('epochs/' + DATA_TYPE + '_' + NET_MODE + '.pth'))
        else:
            model.load_state_dict(torch.load('epochs/' + DATA_TYPE + '_' + NET_MODE + '.pth', map_location='cpu'))

        if torch.cuda.is_available():
            images = images.to('cuda')

        conv1_heat_maps, features_heat_maps = AM_method(images)

        save_image(conv1_heat_maps, filename='vis_%s_%s_%s_conv1.png' % (DATA_TYPE, DATA_MODE, NET_MODE), nrow=nrow,
                   normalize=True, padding=4, pad_value=255)
        save_image(features_heat_maps, filename='vis_%s_%s_%s_features.png' % (DATA_TYPE, DATA_MODE, NET_MODE),
                   nrow=nrow, padding=4, pad_value=255)
