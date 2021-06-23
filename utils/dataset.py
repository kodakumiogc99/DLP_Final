from torch.utils.data import Dataset
from torchvision import transforms
import torch

import os
import glob
from PIL import Image


def train_trans(img):
    trans = transforms.Compose([
        transforms.RandomRotation(degrees=(-179, 179)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return trans(img)


def eval_trans(img):
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return trans(img)


def convert(name):
    animal = {
        'antelope',
        'cow',
        'elephant',
        'lion',
        'squirrel',
        'collie',
        'deer',
        'giraffe',
        'rabbit',
        'zebra'
    }
    for i, n in enumerate(animal):
        if(name == n):
            return i


def AwAData(file: str):
    directory = glob.glob(f'{file}/*')
    img_list = []
    label_list = []

    for d in directory:
        for img in glob.glob(f'{d}/*.jpg'):
            img_list.append(img)
            label_list.append(convert(os.path.basename(d)))

    return img_list, torch.LongTensor(label_list)


class AwASet(Dataset):
    def __init__(self, root_folder, train=True):
        self.img_list, self.label_list = AwAData(root_folder)
        self.train = train

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        img = Image.open(f'{self.img_list[index]}').convert('RGB')
        if self.train:
            img = train_trans(img)
        else:
            img = eval_trans(img)

        label = self.label_list[index]

        return img, label


if __name__ == '__main__':

    AwAData('../dataset')
