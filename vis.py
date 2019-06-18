import argparse
import math

import torch
from torchvision.utils import save_image

from probam import ProbAM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Capsule Network Focused Parts')
    parser.add_argument('--data_name', default='voc', type=str, choices=['voc', 'coco', 'cityscapes'],
                        help='dataset name')
    parser.add_argument('--batch_size', default=64, type=int, help='vis batch size')
    parser.add_argument('--num_iterations', default=3, type=int, help='routing iterations number')

    opt = parser.parse_args()
    DATA_NAME, BATCH_SIZE, NUM_EPOCH = opt.data_name, opt.batch_size, opt.num_epochs
    nrow = math.floor(math.sqrt(BATCH_SIZE))

    transform_test = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
    test_set = ImageFolder(root='data/{}/test'.format(DATA_NAME), transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
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
