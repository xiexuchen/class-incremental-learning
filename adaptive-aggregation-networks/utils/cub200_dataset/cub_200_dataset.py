# @ FileName: cub_200_dataset.py
# @ Author: Alexis
# @ Time: 20-5-16 下午10:10

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
from os.path import join
from PIL import Image

except_list = ["001", "002", "003", "014", "047", "068", "070", "073","076", "090"]
class CUB_200(Dataset):
    def __init__(self, root="./data", train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.classes_name = None
        self.data, self.targets = self.get_data(self.root)
        self.classes = list(range(len(self.classes_name)))
        self.target_img_dict = dict()
        targets = np.array(self.targets)
        for target in self.classes:
            indexes = np.nonzero(targets == target)[0]
            self.target_img_dict.update({target: indexes})

    def get_data(self, data_dir):
        if self.train:
            dataset = torchvision.datasets.ImageFolder(root=join(data_dir, 'train'))
        else:
            dataset = torchvision.datasets.ImageFolder(root=join(self.root, 'test'))
        samples = dataset.samples
        data = []
        target = []
        self.classes_name = dataset.classes
        for item in samples:
            # print(item[1])
            if item[1] not in except_list:
                image = pil_loader(item[0])
                data.append(image)
                target.append(item[1])
        return data, target

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


def pil_loader(path):
    """Image Loader
    """
    with open(path, "rb") as afile:
        img = Image.open(afile)
        return img.convert("RGB")


if __name__ == '__main__':
    dataset = CUB_200(root='../../../open_data_set/CUB_200_2011/dataset', train=True)
    for item in dataset:
        print(item)
    # for key in dataset.target_img_dict.keys():
    #     print(len(dataset.target_img_dict[key]))
