# @ FileName: skin7_augmentation_dataset.py
# @ Author: Alexis
# @ Time: 20-1-17 上午10:53

import os
import sys

import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class Skin7_Augmentation(Dataset):
    """SKin Lesion"""

    def __init__(self, root="./data", train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train

        iter_fold = 1
        self.data, self.targets = self.get_data(iter_fold, self.root)
        self.classes_name = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        self.classes = list(range(len(self.classes_name)))
        self.target_img_dict = dict()
        targets = np.array(self.targets)
        for target in self.classes:
            indexes = np.nonzero(targets == target)[0]
        self.target_img_dict.update({target: indexes})

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.data[index]
        target = self.targets[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_data(self, iterNo, data_dir):

        if self.train:
            # csv = 'split_data/split_data_{}_fold_train.csv'.format(iterNo)
            # csv = 'split_data/rotation_train.csv'
            # csv = 'split_data/add_train.csv'
            csv = 'split_data/main_add_train.csv'
        else:
            # csv = 'split_data/split_data_{}_fold_test.csv'.format(iterNo)
            # csv = 'split_data/rotation_test.csv'
            # csv = 'split_data/add_test.csv'
            csv = 'split_data/main_add_test.csv'
        fn = os.path.join(data_dir, csv)
        print(fn)
        csvfile = pd.read_csv(fn)
        raw_data = csvfile.values

        data = []
        targets = []
        for path, label in raw_data:
            # data.append(os.path.join(self.root,
            #                          "ISIC2018_Task3_Training_Input", path))
            # data.append(os.path.join(self.root,
            #                          "ISIC2018_Task3_Training_Rotation", path))
            data.append(os.path.join(self.root,
                                     "ISIC2018_Task3_Training_Add", path))
            targets.append(label)

        return data, targets


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def pil_loader(path):
    """Image Loader
    """
    with open(path, "rb") as afile:
        img = Image.open(afile)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def print_dataset(dataset, print_time):
    print(len(dataset))
    from collections import Counter
    counter = Counter()
    labels = []
    for index, (img, label) in enumerate(dataset):
        if index % print_time == 0:
            print(img.size(), label)
        labels.append(label)
    counter.update(labels)
    print(counter)


if __name__ == "__main__":
    root = "/home/cccc/Desktop/share/open_data_set/ISIC_2018_Classification"
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = Skin7_Augmentation(root=root, train=True, transform=tf)
    # print_dataset(dataset, print_time=1000)
    for item in dataset:
        print(item[0].shape, item[1])
    # dataset = Skin7(root=root, train=False, transform=transforms.ToTensor())
    # print_dataset(dataset, print_time=1000)
