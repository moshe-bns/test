from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class SimpleCustomData(Dataset):
    def __init__(self, data, targets, transform = None):
        self.targets = targets
        self.data=data
        self.transform = transform
        assert len(data)==len(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target =  self.targets[idx]
        if self.transform is not None:
            target = self.transform(target)
        return self.data[idx], target

class BagsDataset(Dataset):
    def __init__(self, bags, data, targets, transform = None, **kwargs):
        assert len(data) == len(targets)
        self.data = data
        self.transform = transform
        self.targets = targets
        self.bags = bags
        all_idx = []
        for bag in bags:
            all_idx += bag.get_indices_in_data()
        all_idx = list(set(all_idx))
        all_idx.sort()
        new_idx = np.arange(len(all_idx)).tolist()
        self.idx_translator = dict(zip(new_idx, all_idx))

    def get_idx_translator(self):
        return self.idx_translator

    def __len__(self):
        return len(self.idx_translator)

    def __getitem__(self, idx):
        target = self.targets[self.idx_translator[idx]]
        if self.transform is not None:
            target = self.transform(target)
        return self.data[self.idx_translator[idx]], target , int(self.idx_translator[idx])


class BagsDatasetVision(BagsDataset):
    def __init__(self, bags, data, targets, transform=None):
        super().__init__(bags, data, targets, None)
        self.vision_transform = transform

    def __getitem__(self, idx):
        img, target, idx_in_origin_data = super().__getitem__(idx)
        target = int(target)
        img = Image.fromarray(img.numpy(), mode='L')
        if self.vision_transform is not None:
            img = self.vision_transform(img)
        return img, target, idx_in_origin_data