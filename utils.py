from torch.utils.data import Dataset, DataLoader


class MusicData(Dataset):
    def __init__(self, data_type, split):
        self.samples = data_type
        self.split = split

    def __getitem__(self, index):
        sample, target = self.samples[index], self.samples[index]
        return sample, target

    def __len__(self):
        return len(self.samples)


def load_data(data_type, batch_size=32):
    train_set = MusicData(data_type, 'train')
    val_set = MusicData(data_type, 'val')
    test_set = MusicData(data_type, 'test')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
