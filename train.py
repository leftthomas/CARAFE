import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchnet as tnt
from capsule_layer.optim import MultiStepRI
from torch.optim.lr_scheduler import MultiStepLR
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from tqdm import tqdm

import utils
from model import Model


def train():
    model.train()
    train_progress, num_data = tqdm(train_loader), 0
    for img, label in train_progress:
        num_data += img.size(0)
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        out, prob = model(img)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        meter_loss.add(loss.item())
        meter_accuracy.add(out.detach().cpu(), label.detach().cpu())
        meter_map.add(out.detach().cpu(), F.one_hot(label, num_classes=len(train_loader.dataset.classes))
                      .detach().cpu())
        meter_confuse.add(out.detach().cpu(), label.detach().cpu())
        train_progress.set_description('Train Epoch: {}---{}/{} Loss: {:.2f} Top1 Accuracy: {:.2f}% Top5 Accuracy: '
                                       '{:.2f}% mAP: {:.2f}%'.format(epoch, num_data, len(train_loader.dataset),
                                                                     meter_loss.value()[0], meter_accuracy
                                                                     .value()[0], meter_accuracy.value()[1],
                                                                     meter_map.value() * 100.0))
    loss_logger.log(epoch, meter_loss.value()[0], name='train')
    accuracy_logger.log(epoch, meter_accuracy.value()[0], name='train_top1')
    accuracy_logger.log(epoch, meter_accuracy.value()[1], name='train_top5')
    map_logger.log(epoch, meter_map.value() * 100.0, name='train')
    train_confuse_logger.log(meter_confuse.value())
    results['train_loss'].append(meter_loss.value()[0])
    results['train_accuracy_1'].append(meter_accuracy.value()[0])
    results['train_accuracy_5'].append(meter_accuracy.value()[1])
    results['train_map'].append(meter_map.value() * 100.0)
    lr_scheduler.step()
    meter_loss.reset()
    meter_accuracy.reset()
    meter_map.reset()
    meter_confuse.reset()


def test():
    model.eval()
    test_progress, num_data = tqdm(test_loader), 0
    for img, label in test_progress:
        num_data += img.size(0)
        img, label = img.to(device), label.to(device)
        out, prob = model(img)
        loss = criterion(out, label)
        meter_loss.add(loss.item())
        meter_accuracy.add(out.detach().cpu(), label.detach().cpu())
        meter_map.add(out.detach().cpu(), F.one_hot(label, num_classes=len(train_loader.dataset.classes))
                      .detach().cpu())
        meter_confuse.add(out.detach().cpu(), label.detach().cpu())
        test_progress.set_description('Test Epoch: {}---{}/{} Loss: {:.2f} Top1 Accuracy: {:.2f}% Top5 Accuracy'
                                      ': {:.2f}% mAP: {:.2f}%'.format(epoch, num_data, len(test_loader.dataset),
                                                                      meter_loss.value()[0], meter_accuracy
                                                                      .value()[0], meter_accuracy.value()[1],
                                                                      meter_map.value() * 100.0))
    loss_logger.log(epoch, meter_loss.value()[0], name='test')
    accuracy_logger.log(epoch, meter_accuracy.value()[0], name='test_top1')
    accuracy_logger.log(epoch, meter_accuracy.value()[1], name='test_top5')
    map_logger.log(epoch, meter_map.value() * 100.0, name='test')
    test_confuse_logger.log(meter_confuse.value())
    results['test_loss'].append(meter_loss.value()[0])
    results['test_accuracy_1'].append(meter_accuracy.value()[0])
    results['test_accuracy_5'].append(meter_accuracy.value()[1])
    results['test_map'].append(meter_map.value() * 100.0)
    global best_acc
    if meter_accuracy.value()[0] > best_acc:
        best_acc = meter_accuracy.value()[0]
        # save model
        torch.save(model.state_dict(), 'epochs/{}.pth'.format(DATA_NAME))
    meter_loss.reset()
    meter_accuracy.reset()
    meter_map.reset()
    meter_confuse.reset()


def vis():
    # generate vis results
    probam = utils.ProbAM(model)
    # for train image
    train_images, _ = next(iter(train_loader))
    train_images = train_images[:16]
    train_original_logger.log(make_grid(train_images, nrow=4, padding=4).numpy())
    train_images = train_images.to(device)
    train_heat_maps, train_cams = probam(train_images)
    train_heatmaps_logger.log(make_grid(train_heat_maps, nrow=4, padding=4).numpy())
    train_cams_logger.log(make_grid(train_cams, nrow=4, padding=4).numpy())
    # for test image
    test_images, _ = next(iter(test_loader))
    test_images = test_images[:16]
    test_original_logger.log(make_grid(test_images, nrow=4, padding=4).numpy())
    test_images = test_images.to(device)
    test_heat_maps, test_cams = probam(test_images)
    test_heatmaps_logger.log(make_grid(test_heat_maps, nrow=4, padding=4).numpy())
    test_cams_logger.log(make_grid(test_cams, nrow=4, padding=4).numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Detection Model')
    parser.add_argument('--data_name', default='voc', type=str, choices=['voc', 'coco', 'cityscapes'],
                        help='dataset name')
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--num_iterations', default=3, type=int, help='routing iterations number')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

    opt = parser.parse_args()
    DATA_NAME, BATCH_SIZE, NUM_ITERATIONS, NUM_EPOCH = opt.data_name, opt.batch_size, opt.num_iterations, opt.num_epochs

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {'train_loss': [], 'test_loss': [], 'train_accuracy_1': [], 'test_accuracy_1': [], 'train_accuracy_5': [],
               'test_accuracy_5': [], 'train_map': [], 'test_map': []}

    # Data
    print('==> Preparing data..')
    train_loader = utils.load_data(DATA_NAME, 'train', BATCH_SIZE, shuffle=True)
    test_loader = utils.load_data(DATA_NAME, 'test', BATCH_SIZE, shuffle=True)

    # Model
    print('==> Building model..')
    model = Model(len(train_loader.dataset.classes), NUM_EPOCH).to(device)
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optim_configs = [{'params': model.features.parameters(), 'lr': 1e-4 * 10},
                     {'params': model.classifier.parameters(), 'lr': 1e-4}]
    optimizer = optim.Adam(optim_configs, lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(NUM_EPOCH * 0.5), int(NUM_EPOCH * 0.7)], gamma=0.1)
    iter_scheduler = MultiStepRI(model, milestones=[int(NUM_EPOCH * 0.7), int(NUM_EPOCH * 0.9)], verbose=True)

    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    meter_map = tnt.meter.mAPMeter()
    meter_confuse = tnt.meter.ConfusionMeter(len(train_loader.dataset.classes), normalized=True)
    loss_logger = VisdomPlotLogger('line', env=DATA_NAME, opts={'title': 'Loss'})
    accuracy_logger = VisdomPlotLogger('line', env=DATA_NAME, opts={'title': 'Accuracy'})
    map_logger = VisdomPlotLogger('line', env=DATA_NAME, opts={'title': 'mAP'})
    train_confuse_logger = VisdomLogger('heatmap', env=DATA_NAME, opts={'title': 'Train Confusion Matrix',
                                                                        'columnnames': train_loader.dataset.classes,
                                                                        'rownames': train_loader.dataset.classes})
    test_confuse_logger = VisdomLogger('heatmap', env=DATA_NAME, opts={'title': 'Test Confusion Matrix',
                                                                       'columnnames': test_loader.dataset.classes,
                                                                       'rownames': test_loader.dataset.classes})
    train_original_logger = VisdomLogger('image', env=DATA_NAME,
                                         opts={'title': 'Train Original Images', 'width': 372, 'height': 372})
    train_heatmaps_logger = VisdomLogger('image', env=DATA_NAME,
                                         opts={'title': 'Train Features Heatmap', 'width': 372, 'height': 372})
    train_cams_logger = VisdomLogger('image', env=DATA_NAME,
                                     opts={'title': 'Train Features CAM', 'width': 372, 'height': 372})
    test_original_logger = VisdomLogger('image', env=DATA_NAME,
                                        opts={'title': 'Test Original Images', 'width': 372, 'height': 372})
    test_heatmaps_logger = VisdomLogger('image', env=DATA_NAME,
                                        opts={'title': 'Test Features Heatmap', 'width': 372, 'height': 372})
    test_cams_logger = VisdomLogger('image', env=DATA_NAME,
                                    opts={'title': 'Test Features CAM', 'width': 372, 'height': 372})

    best_acc = 0
    for epoch in range(1, NUM_EPOCH + 1):
        # train loop
        train()
        # test loop
        with torch.no_grad():
            test()
            vis()
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('statistics/{}_results.csv'.format(DATA_NAME), index_label='epoch')
        iter_scheduler.step()
