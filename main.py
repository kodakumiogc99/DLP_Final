from utils.dataset import AwASet
from utils.train import train, test

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import models

import numpy
import argparse
import os
import shutil


if __name__ == '__main__':

    torch.cuda.empty_cache()


    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--model', type=str, default='vgg19')

    args = parser.parse_args()

    if not os.path.exists(f'{args.checkpoint}'):
        os.mkdir(f'{args.checkpoint}')

    trainset = AwASet('./dataset')
    ratio = int(len(trainset) * 0.9)
    trainset, testset = random_split(trainset, (ratio, len(trainset)-ratio))
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False)

    net = models.vgg19(pretrained=True)

    net.fc = nn.Linear(1000, 10)


    net = net.to(args.device)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    loss_func = nn.CrossEntropyLoss()

    train(args, trainloader,testloader, net, optim, loss_func)
