import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Detection Model')
    parser.add_argument('--data_name', default='voc', type=str, choices=['voc', 'coco', 'cityscapes'],
                        help='dataset name')
    parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

    opt = parser.parse_args()
    DATA_NAME, BATCH_SIZE, NUM_EPOCH = opt.data_name, opt.batch_size, opt.num_epochs

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {'train_loss': [], 'test_loss': [], 'train_accuracy_1': [], 'test_accuracy_1': [], 'train_accuracy_5': [],
               'test_accuracy_5': []}

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([transforms.Resize(224), transforms.RandomCrop(224), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
    train_set = ImageFolder(root='data/{}/train'.format(DATA_NAME), transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    test_set = ImageFolder(root='data/{}/test'.format(DATA_NAME), transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

    # Model
    print('==> Building model..')
    model = Model(len(train_set.classes)).to(device)
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    meter_confuse = tnt.meter.ConfusionMeter(len(train_set.classes), normalized=True)
    loss_logger = VisdomPlotLogger('line', env=DATA_NAME, opts={'title': 'Loss'})
    accuracy_logger = VisdomPlotLogger('line', env=DATA_NAME, opts={'title': 'Accuracy'})
    train_confuse_logger = VisdomLogger('heatmap', env=DATA_NAME, opts={'title': 'Train Confusion Matrix',
                                                                        'columnnames': train_set.classes,
                                                                        'rownames': train_set.classes})
    test_confuse_logger = VisdomLogger('heatmap', env=DATA_NAME, opts={'title': 'Test Confusion Matrix',
                                                                       'columnnames': test_set.classes,
                                                                       'rownames': test_set.classes})

    for epoch in range(1, NUM_EPOCH + 1):
        # train loop
        model.train()
        train_progress, num_data = tqdm(train_loader), 0
        for img, label in train_progress:
            num_data += img.size(0)
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            meter_loss.add(loss.item())
            meter_accuracy.add(out.detach().cpu(), label.detach().cpu())
            meter_confuse.add(out.detach().cpu(), label.detach().cpu())
            train_progress.set_description('Train Epoch: {}---{}/{} Loss: {:.2f} Top1 Accuracy: {:.2f}% Top5 Accuracy: '
                                           '{:.2f}%'.format(epoch, num_data, len(train_set), meter_loss.value()[0],
                                                            meter_accuracy.value()[0], meter_accuracy.value()[1]))
        loss_logger.log(epoch, meter_loss.value()[0], name='train')
        accuracy_logger.log(epoch, meter_accuracy.value()[0], name='train_top1')
        accuracy_logger.log(epoch, meter_accuracy.value()[1], name='train_top5')
        train_confuse_logger.log(meter_confuse.value())
        results['train_loss'].append(meter_loss.value()[0])
        results['train_accuracy_1'].append(meter_accuracy.value()[0])
        results['train_accuracy_5'].append(meter_accuracy.value()[1])
        meter_loss.reset()
        meter_accuracy.reset()
        meter_confuse.reset()

        # test loop
        with torch.no_grad():
            model.eval()
            test_progress, num_data = tqdm(test_loader), 0
            for img, label in test_progress:
                num_data += img.size(0)
                img, label = img.to(device), label.to(device)
                out = model(img.to(device))
                loss = criterion(out, label)
                meter_loss.add(loss.item())
                meter_accuracy.add(out.detach().cpu(), label.detach().cpu())
                meter_confuse.add(out.detach().cpu(), label.detach().cpu())
                test_progress.set_description('Test Epoch: {}---{}/{} Loss: {:.2f} Top1 Accuracy: {:.2f}% Top5 '
                                              'Accuracy: {:.2f}%'.format(epoch, num_data, len(test_set),
                                                                         meter_loss.value()[0], meter_accuracy.
                                                                         value()[0], meter_accuracy.value()[1]))
            loss_logger.log(epoch, meter_loss.value()[0], name='test')
            accuracy_logger.log(epoch, meter_accuracy.value()[0], name='test_top1')
            accuracy_logger.log(epoch, meter_accuracy.value()[1], name='test_top5')
            test_confuse_logger.log(meter_confuse.value())
            results['test_loss'].append(meter_loss.value()[0])
            results['test_accuracy_1'].append(meter_accuracy.value()[0])
            results['test_accuracy_5'].append(meter_accuracy.value()[1])
            meter_loss.reset()
            meter_accuracy.reset()
            meter_confuse.reset()

        # save model
        torch.save(model.state_dict(), 'epochs/{}_{}.pth'.format(DATA_NAME, epoch))
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('statistics/{}_results.csv'.format(DATA_NAME), index_label='epoch')
