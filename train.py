import argparse

import pandas as pd
import torch
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
    for imgs, boxes, labels in train_progress:
        num_data += imgs.size(0)
        labels = utils.creat_multi_label(labels, utils.num_classes[DATA_NAME])
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out, prob = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        meter_loss.add(loss.item())
        meter_map.add(out.detach().cpu(), labels.detach().cpu())
        train_progress.set_description('Train Epoch: {}---{}/{} Loss: {:.4f} mAP: {:.2f}%'
                                       .format(epoch, num_data, len(train_loader.dataset), meter_loss.value()[0],
                                               meter_map.value() * 100.0))
    loss_logger.log(epoch, meter_loss.value()[0], name='train')
    map_logger.log(epoch, meter_map.value() * 100.0, name='train')
    results['train_loss'].append(meter_loss.value()[0])
    results['train_map'].append(meter_map.value() * 100.0)
    lr_scheduler.step(epoch)
    meter_loss.reset()
    meter_map.reset()


def val():
    model.eval()
    val_progress, num_data = tqdm(val_loader), 0
    for imgs, boxes, labels in val_progress:
        num_data += imgs.size(0)
        labels = utils.creat_multi_label(labels, utils.num_classes[DATA_NAME])
        imgs, labels = imgs.to(device), labels.to(device)
        out, prob = model(imgs)
        loss = criterion(out, labels)
        meter_loss.add(loss.item())
        meter_map.add(out.detach().cpu(), labels.detach().cpu())
        val_progress.set_description('Val Epoch: {}---{}/{} Loss: {:.4f} mAP: {:.2f}%'
                                     .format(epoch, num_data, len(val_loader.dataset), meter_loss.value()[0],
                                             meter_map.value() * 100.0))
    loss_logger.log(epoch, meter_loss.value()[0], name='val')
    map_logger.log(epoch, meter_map.value() * 100.0, name='val')
    results['val_loss'].append(meter_loss.value()[0])
    results['val_map'].append(meter_map.value() * 100.0)
    global best_map
    if meter_map.value() > best_map:
        best_map = meter_map.value()
        # save model
        torch.save(model.state_dict(), 'epochs/{}.pth'.format(DATA_NAME))
    meter_loss.reset()
    meter_map.reset()


def vis():
    # generate vis results
    probam = utils.ProbAM(model)
    # for train image
    train_images, train_boxes, train_labels = next(iter(train_loader))
    train_images = train_images[:16]
    train_original_logger.log(make_grid(train_images, nrow=4, padding=4, normalize=True).numpy())
    train_images = train_images.to(device)
    train_heat_maps, train_cams = probam(train_images)
    train_heatmaps_logger.log(make_grid(train_heat_maps, nrow=4, padding=4, normalize=True).numpy())
    train_cams_logger.log(make_grid(train_cams, nrow=4, padding=4, normalize=True).numpy())
    # for val image
    val_images, val_boxes, val_labels = next(iter(val_loader))
    val_images = val_images[:16]
    val_original_logger.log(make_grid(val_images, nrow=4, padding=4, normalize=True).numpy())
    val_images = val_images.to(device)
    val_heat_maps, val_cams = probam(val_images)
    val_heatmaps_logger.log(make_grid(val_heat_maps, nrow=4, padding=4, normalize=True).numpy())
    val_cams_logger.log(make_grid(val_cams, nrow=4, padding=4, normalize=True).numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Detection Model')
    parser.add_argument('--data_name', default='voc', type=str, choices=['voc', 'coco'], help='dataset name')
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--num_iterations', default=3, type=int, help='routing iterations number')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

    opt = parser.parse_args()
    DATA_NAME, BATCH_SIZE, NUM_ITERATIONS, NUM_EPOCH = opt.data_name, opt.batch_size, opt.num_iterations, opt.num_epochs

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {'train_loss': [], 'val_loss': [], 'train_map': [], 'val_map': []}

    print('==> Preparing data..')
    train_loader = utils.load_data(DATA_NAME, 'train', BATCH_SIZE, shuffle=True)
    val_loader = utils.load_data(DATA_NAME, 'val', BATCH_SIZE, shuffle=True)

    print('==> Building model..')
    model = Model(utils.num_classes[DATA_NAME], NUM_ITERATIONS).to(device)
    print("# trainable parameters:", sum(param.numel() if param.requires_grad else 0 for param in model.parameters()))
    criterion = utils.MarginLoss()
    optim_configs = [{'params': model.features.parameters(), 'lr': 1e-4 * 10},
                     {'params': model.classifier.parameters(), 'lr': 1e-4}]
    optimizer = optim.Adam(optim_configs, lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(NUM_EPOCH * 0.5), int(NUM_EPOCH * 0.7)], gamma=0.1)
    iter_scheduler = MultiStepRI(model, milestones=[int(NUM_EPOCH * 0.7), int(NUM_EPOCH * 0.9)], verbose=True)

    meter_loss, meter_map = tnt.meter.AverageValueMeter(), tnt.meter.mAPMeter()
    loss_logger = VisdomPlotLogger('line', env=DATA_NAME, opts={'title': 'Loss', 'width': 350, 'height': 350})
    map_logger = VisdomPlotLogger('line', env=DATA_NAME, opts={'title': 'mAP', 'width': 350, 'height': 350})
    train_original_logger = VisdomLogger('image', env=DATA_NAME,
                                         opts={'title': 'Train Original Images', 'width': 350, 'height': 350})
    train_heatmaps_logger = VisdomLogger('image', env=DATA_NAME,
                                         opts={'title': 'Train Features Heatmap', 'width': 350, 'height': 350})
    train_cams_logger = VisdomLogger('image', env=DATA_NAME,
                                     opts={'title': 'Train Features CAM', 'width': 350, 'height': 350})
    val_original_logger = VisdomLogger('image', env=DATA_NAME,
                                       opts={'title': 'Val Original Images', 'width': 350, 'height': 350})
    val_heatmaps_logger = VisdomLogger('image', env=DATA_NAME,
                                       opts={'title': 'Val Features Heatmap', 'width': 350, 'height': 350})
    val_cams_logger = VisdomLogger('image', env=DATA_NAME,
                                   opts={'title': 'Val Features CAM', 'width': 350, 'height': 350})

    best_map = 0
    for epoch in range(1, NUM_EPOCH + 1):
        train()
        with torch.no_grad():
            val()
            vis()
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('statistics/{}_results.csv'.format(DATA_NAME), index_label='epoch')
        iter_scheduler.step()
