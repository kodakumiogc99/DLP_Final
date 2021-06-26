import os
import random
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from torchvision import models

from utils.dataset import AwASet
from utils.train import train, test


warnings.filterwarnings('ignore', category=UserWarning)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_folder', type=str, default='./Dataset')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--model', type=str, default='vgg16')
    parser.add_argument('--load_target', type=str, default=None)
    parser.add_argument('--train_target', action='store_true')

    args = parser.parse_args()

    print('=' * 100)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('=' * 100)

    return args


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

    args = parse()

    if not os.path.exists(f'{args.checkpoint}'):
        os.mkdir(f'{args.checkpoint}')

    trainset = AwASet(args.root_folder)

    ratio = int(len(trainset) * 0.9)
    trainset, testset = random_split(trainset, (ratio, len(trainset) - ratio))

    testset.dataset.__setattr__('train', False)

    num_workers = len(os.sched_getaffinity(0))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=num_workers, shuffle=False)

    net = models.vgg16(pretrained=True)
    net.classifier[-1] = nn.Linear(4096, 10)

    if args.load_target:
        net = torch.load(args.load_target)

    net = net.to(args.device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    if args.train_target:
        train(args, trainloader, testloader, net, optimizer, loss_func)

    test(args, testloader, net, loss_func)
