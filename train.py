import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from thop import profile, clever_format
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Model
from utils import ImageReader, recall


def train(net, optim):
    net.train()
    total_loss, total_correct, total_num, data_bar = 0, 0, 0, tqdm(train_data_loader)
    for inputs, labels in data_bar:
        inputs, labels = inputs.cuda(), labels.cuda()
        features, out = net(inputs)
        loss = cel_criterion(out.permute(0, 2, 1).contiguous(), labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        pred = torch.argmax(out, dim=-1)
        total_loss += loss.item()
        total_correct += torch.sum(pred == labels).item() / ENSEMBLE_SIZE
        total_num += inputs.size(0)
        data_bar.set_description('Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
                                 .format(epoch, num_epochs, total_loss / total_num, total_correct / total_num * 100))

    return total_loss / total_num, total_correct / total_num * 100


def eval(net, recalls):
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        for key in eval_dict.keys():
            eval_dict[key]['features'] = []
            for inputs, labels in tqdm(eval_dict[key]['data_loader'], desc='processing {} data'.format(key)):
                inputs, labels = inputs.cuda(), labels.cuda()
                features, out = net(inputs)
                features = F.normalize(torch.sum(features, dim=1), dim=-1)
                eval_dict[key]['features'].append(features)
            eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)

    # compute recall metric
    if data_name == 'isc':
        acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recalls,
                          eval_dict['gallery']['features'], gallery_data_set.labels)
    else:
        acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recalls)
    desc = ''
    for index, id in enumerate(recalls):
        desc += 'R@{}:{:.2f}% '.format(id, acc_list[index] * 100)
        results['test_recall@{}'.format(recall_ids[index])].append(acc_list[index] * 100)
    print(desc)
    return acc_list[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Retrieval Model')
    parser.add_argument('--data_path', default='/home/data', type=str, help='datasets path')
    parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'sop', 'isc'],
                        help='dataset name')
    parser.add_argument('--crop_type', default='uncropped', type=str, choices=['uncropped', 'cropped'],
                        help='crop data or not, it only works for car or cub dataset')
    parser.add_argument('--backbone_type', default='resnet18', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnext50'], help='backbone type')
    parser.add_argument('--feature_dim', default=128, type=int, help='feature dim')
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature used in softmax')
    parser.add_argument('--batch_size', default=512, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=500, type=int, help='train epoch number')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')

    opt = parser.parse_args()
    # args parse
    data_path, data_name, crop_type, feature_dim = opt.data_path, opt.data_name, opt.crop_type, opt.feature_dim
    temperature, batch_size, num_epochs, recalls = opt.temperature, opt.batch_size, opt.num_epochs, opt.recalls
    backbone_type = opt.backbone_type
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(data_name, crop_type, backbone_type, feature_dim, temperature,
                                                  batch_size, num_epochs)
    recall_ids = [int(k) for k in recalls.split(',')]

    results = {'train_loss': []}
    for r in recall_ids:
        results['test_recall@{}'.format(r)] = []

    # dataset loaders
    train_data_set = ImageReader(data_path, data_name, 'train', crop_type)
    train_data_loader = DataLoader(train_data_set, batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                   drop_last=True)
    train_ext_data_set = ImageReader(data_path, data_name, 'train_ext', crop_type)
    train_ext_data_loader = DataLoader(train_ext_data_set, batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data_set = ImageReader(data_path, data_name, 'query' if data_name == 'isc' else 'test', crop_type)
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=16, pin_memory=True)
    eval_dict = {'train': {'data_loader': train_ext_data_loader}, 'test': {'data_loader': test_data_loader}}
    if data_name == 'isc':
        gallery_data_set = ImageReader(data_path, data_name, 'gallery', crop_type)
        gallery_data_loader = DataLoader(gallery_data_set, batch_size, shuffle=False, num_workers=16, pin_memory=True)
        eval_dict['gallery'] = {'data_loader': gallery_data_loader}

    # model setup, model profile and optimizer config
    model = Model(backbone_type, feature_dim).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = Adam(model.parameters(), lr=1e-3)

    # train and eval loop
    best_recall = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, optimizer)
        results['train_loss'].append(train_loss)
        rank = eval(model, recall_ids)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        # save database and model
        data_base = {}
        if rank > best_recall:
            best_recall = rank
            data_base['train_images'] = train_ext_data_set.images
            data_base['train_labels'] = train_ext_data_set.labels
            data_base['train_features'] = eval_dict['train']['features']
            data_base['test_images'] = test_data_set.images
            data_base['test_labels'] = test_data_set.labels
            data_base['test_features'] = eval_dict['test']['features']
            if data_name == 'isc':
                data_base['gallery_images'] = gallery_data_set.images
                data_base['gallery_labels'] = gallery_data_set.labels
                data_base['gallery_features'] = eval_dict['gallery']['features']
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
            torch.save(data_base, 'results/{}_data_base.pth'.format(save_name_pre))
