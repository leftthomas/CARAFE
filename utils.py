import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder


def load_data(data_name, data_type, batch_size):
    if data_type == 'train':
        transform = transforms.Compose([transforms.Resize(224), transforms.RandomCrop(224), transforms.ToTensor()])
        shuffle = True
    else:
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
        shuffle = False
    data_set = ImageFolder(root='data/{}/{}'.format(data_name, data_type), transform=transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=16)
    return data_loader
