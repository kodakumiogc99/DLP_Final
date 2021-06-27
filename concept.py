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

from utils.dataset import AwASet
from utils.train import train_concept, test_concept

warnings.filterwarnings('ignore', category=UserWarning)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, required=True)
    parser.add_argument('--load_target', type=str, required=True)
    parser.add_argument('--train_concept', action='store_true')
    parser.add_argument('--load_concept_v', type=str, default=None)
    parser.add_argument('--load_concept_g', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--accuracy_threshold', type=float, default=0.97)
    parser.add_argument('--num_concepts', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
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

    train_set = AwASet(args.root_folder)

    ratio = int(len(train_set) * 0.9)
    train_set, test_set = random_split(train_set, (ratio, len(train_set) - ratio))

    test_set.dataset.__setattr__('train', False)

    num_workers = len(os.sched_getaffinity(0))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=num_workers, shuffle=False)

    net = torch.load(args.load_target)

    phi = nn.Sequential(
        net.features,
        net.avgpool
    )

    h = nn.Sequential(
        net.classifier
    )

    phi = phi.to(args.device)
    h = h.to(args.device)

    phi.eval()
    for paramter in phi.parameters():
        paramter.requires_grad = False

    h.eval()
    for paramter in h.parameters():
        paramter.requires_grad = False

    latent_dim = args.latent_dim
    num_concepts = args.num_concepts

    v = nn.Linear(latent_dim, num_concepts, bias=False)

    g = nn.Sequential(
        nn.Linear(num_concepts, latent_dim * 2, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(latent_dim * 2, latent_dim, bias=False),
        nn.ReLU(inplace=True)
    )

    if args.load_concept_v:
        v = torch.load(args.load_concept_v)

    if args.load_concept_g:
        g = torch.load(args.load_concept_g)

    v = v.to(args.device)
    g = g.to(args.device)

    v_optimizer = optim.Adam(v.parameters(), lr=args.lr)
    g_optimizer = optim.Adam(g.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    if args.train_concept:
        train_concept(args, phi, h, v, g, train_loader, test_loader, v_optimizer, g_optimizer, criterion)

    test_concept(args, phi, h, v, g, test_loader, criterion, save_result=True)
