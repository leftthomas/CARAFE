from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class VideoData(Dataset):
    def __init__(self, path, t):
        self.samples = path
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.samples[index]
        return self.T(sample), target

    def __len__(self):
        return len(self.samples)


def load_data(batch_size=32):
    train_set = VideoData('data/train', transforms.ToTensor())
    test_set = VideoData('data/test', transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
