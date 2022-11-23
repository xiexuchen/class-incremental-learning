import os
import sys

import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class SD_198(Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train

        iter_fold = 1

        split_mode = 1

        self.data, self.targets = self.getdata(iter_fold, self.root, split_mode)
        if split_mode == 0:
            self.classes_name = os.listdir(os.path.join(root, 'images'))
            self.classes = list(range(len(self.classes_name)))
            self.target_img_dict = dict()
            targets = np.array(list(map(int, self.targets)))
            #把找出每一类当前所有的下标
            for target in self.classes:
                indexes = np.nonzero(targets == target)[0]
                self.target_img_dict.update({target: indexes})

    def getdata(self, iterNo, data_dir, split_mode):
        if self.train:
            if split_mode == 0:
                txt = 'other_classes_split/train_{}.txt'.format(iterNo)
            elif split_mode == 1:
                txt = 'main_classes_split/train_{}.txt'.format(iterNo)
            elif split_mode == 2:
                txt = 'balanced_other_classes_split/train_{}.txt'.format(iterNo)
            elif split_mode == 3:
                txt = 'incremental_label/incremental_train_{}.txt'.format(iterNo)
            elif split_mode == 4:
                txt = 'sd_193_pretrain_incremental_label/incremental_train_{}.txt'.format(iterNo)
            elif split_mode == 5:
                txt = 'sd_158_pretrain_incremental_label/incremental_train_{}.txt'.format(iterNo)
            elif split_mode == 6:
                txt = 'sd_158_pretrain_incremental_label_change/incremental_train_{}.txt'.format(iterNo)
            else:
                raise ValueError('No split model:{}'.format(split_mode))
        else:
            if split_mode == 0:
                txt = 'other_classes_split/val_{}.txt'.format(iterNo)
            elif split_mode == 1:
                txt = 'main_classes_split/val_{}.txt'.format(iterNo)
            elif split_mode == 2:
                txt = 'balanced_other_classes_split/val_{}.txt'.format(iterNo)
            elif split_mode == 3:
                txt = 'incremental_label/incremental_val_{}.txt'.format(iterNo)
            elif split_mode == 4:
                txt = 'sd_193_pretrain_incremental_label/incremental_val_{}.txt'.format(iterNo)
            elif split_mode == 5:
                txt = 'sd_158_pretrain_incremental_label/incremental_val_{}.txt'.format(iterNo)
            elif split_mode == 6:
                txt = 'sd_158_pretrain_incremental_label_change/incremental_train_{}.txt'.format(iterNo)
            else:
                raise ValueError('No split model:{}'.format(split_mode))

        fn = os.path.join(data_dir, txt)
        print(fn)
        file = open(fn)
        file_name_list = file.read().split('\n')
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(' ')
            if len(temp) == 2:
                path = os.path.join(self.root, 'images', temp[0])
                image = pil_loader(path)
                data.append(image)                
                targets.append(int(temp[1]))
        return data, targets

    def __getitem__(self, idx):
        path = self.data[idx]
        target = self.targets[idx]
        img = pil_loader(path)
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
    trainset = SD_198(root=os.envrion["SD198DATASET"], train=True, transform=None)
    print(len(trainset))
    # for item in trainset:
    #     print(item[0], item[1])
