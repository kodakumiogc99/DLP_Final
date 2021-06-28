import os
import random
import argparse
import warnings
import copy
import numpy as np
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid, save_image

from utils.dataset import AwASet
from utils.train import train_concept, test_concept
from utils.reverse import ReverseVGG16

warnings.filterwarnings('ignore', category=UserWarning)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=str, default='checkpoint', help='path of model')
    parser.add_argument('--decoder', type=str, default='decoder_135', help='decoder name, decoder_135')
    parser.add_argument('--vector', type=str, default='v_8', help='vector name, v_8')
    parser.add_argument('--visual_dir', type=str, default='visualize', help='vector name, v_8')
    args = parser.parse_args()


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

    decoder = torch.load(f'{args.checkpoint}/{args.decoder}.pt')
    decoder = nn.DataParallel(decoder, device_ids=[0])

    decoder = decoder.to(args.device)

    concept = torch.load(f'{args.checkpoint}/{args.vector}.pt')
    if os.path.exists(args.visual_dir):
        shutil.rmtree(args.visual_dir)
    os.mkdir(args.visual_dir)

    concept = concept.weight.data * 255
    concept = F.relu(concept)
    concept = concept.to(args.device)

    for i in range(8):
        visual = concept[i, :].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        visual = decoder(visual)
        image = make_grid(visual, normalize=True)
        save_image(image, f'{args.visual_dir}/{i}.jpg')
