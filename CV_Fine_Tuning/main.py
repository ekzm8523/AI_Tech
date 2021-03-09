
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg11
from torch.optim import Adam

import os
import cv2
import numpy as np
from glob import glob


class MaskDataset(Dataset):
    def __init__(self, data_root, is_train=True, input_size=224, transform=None):
        super(MaskDataset, self).__init__()

        self.img_list = self._load_img_list(data_root, is_train)
        self.len = len(self.img_list)
        self.input_size = input_size
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_list[index]

        # Image Loading
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img /= 255

        if self.transform:
            img = self.transform(img)

        # Ground Truth
        label = self._get_class_idx_from_img_name(img_path)

        return img, label

    def __len__(self):
        return self.len

    def _load_img_list(self, data_root, is_train):
        # Change the name of directory which has inconsistent naming rule.
        full_img_list = glob(data_root + '/*')
        for dir in full_img_list:
            dirname = os.path.basename(dir)
            if '-1' in dirname:
                os.rename(dir, dir.replace(dirname, dirname.replace('-1', '1')))
        # ID < 1000 for Training (N=721)
        # 1000 < ID < 1050 for Validation (N=63)
        img_list = []
        for dir in glob(data_root + '/*V'):
            if is_train and (self._load_img_id(dir) < 500):
                img_list.extend(glob(dir + '/*'))
            elif not is_train and (1000 < self._load_img_id(dir) < 1050):
                img_list.extend(glob(dir + '/*'))
        return img_list


    def _load_img_id(self, img_path):
        return int(os.path.basename(img_path).split('_')[0])

    def _get_class_idx_from_img_name(self, img_path):
        img_name = os.path.basename(img_path)

        if 'normal' in img_name: return 0
        elif 'mask1' in img_name: return 1
        elif 'mask2' in img_name: return 2
        elif 'mask3' in img_name: return 3
        elif 'mask4' in img_name: return 4
        elif 'mask5' in img_name: return 5
        elif 'incorrect_mask' in img_name: return 6
        else:
            raise ValueError("%s is not a valid filename. Please change the name of %s." % (img_name, img_path))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    

def get_data(data_root, batch_size, input_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MaskDataset(data_root, is_train=True, input_size=input_size, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)

    valid_dataset = MaskDataset(data_root, is_train=False, input_size=input_size, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    return train_loader, valid_loader

if __name__ == '__main__':
    # parameter setting
    data_root = '/content/gdrive/MyDrive/FaceDataset/data'
    log_dir   = '/content/gdrive/MyDrive/FaceDataset/log'
    batch_size = 8
    lr = 1e-4
    input_size = 224


    pretrained = True

    model = vgg11(pretrained)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=7, bias=True)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'scratch_train_log.csv'), 'w') as log:
        for iter, ()



    
