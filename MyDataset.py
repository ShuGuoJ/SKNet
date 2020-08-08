from torchvision import transforms
import os, torch
from PIL import Image
from torch.utils.data import Dataset
import random
from random import shuffle
'''自定义数据集'''

class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        super(MyDataset, self).__init__()
        self.datas = os.listdir(path)
        self.root = path
        self.transform = transform

        # shuffle(self.datas)

    def __len__(self):
        return len(self.datas)


    def __getitem__(self, index):
        label = self.datas[index].split('.')[0]
        label = 1 if label=='dog' else 0
        path = os.path.join(self.root, self.datas[index])
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)

