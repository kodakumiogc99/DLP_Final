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
    parser.add_argument('--root_folder', type=str, default='./Dataset')
    parser.add_argument('--load_target', type=str)
    parser.add_argument('--train_concept', action='store_true')
    parser.add_argument('--load_concept_v', type=str, default=None)
    parser.add_argument('--load_concept_g', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--accuracy_threshold', type=float, default=0.97)
    parser.add_argument('--num_concepts', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=7e-4)
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
    test_set.dataset = copy.deepcopy(train_set.dataset)

    test_set.dataset.__setattr__('train', False)

    num_workers = len(os.sched_getaffinity(0))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=num_workers, shuffle=False)

    net = torch.load('checkpoint/vgg16_10.pt')

    # decoder = ReverseVGG16()
    decoder = torch.load('checkpoint/decoder_49.pt')
    decoder = nn.DataParallel(decoder,device_ids=[0])

    decoder = decoder.to(args.device)

    phi = nn.Sequential(
        net.features,
        net.avgpool
    )

    phi = nn.DataParallel(phi)

    phi = phi.to(args.device)

    decoder_optim = optim.Adam(decoder.parameters(), lr=args.lr)

    loss_func = nn.MSELoss()

    phi.eval()
    for paramter in phi.parameters():
        paramter.requires_grad = False

    # if os.path.exists('result_images'):
        # shutil.rmtree('result_images')
    # os.mkdir('result_images')

    epoch_length = len(str(args.epochs))

    scheduler = optim.lr_scheduler.StepLR(decoder_optim, 1, 0.99)

    for epoch in range(50, args.epochs):

        total_loss = 0.0
        last_length = 0

        if not os.path.exists(f'result_images/{epoch}'):
            os.mkdir(f'result_images/{epoch}')

        for batch_index, (img, label) in enumerate(train_loader):
            img = img.to(args.device)
            label = label.to(args.device)

            x = phi(img)

            batch_loss = torch.tensor([0.0]).to(args.device)

            padding_image = F.pad(img, (90, 90, 90, 90), 'constant', -1)
            """
            NCHW
            """
            decoder_optim.zero_grad()

            for i in range(x.shape[2]):
                for j in range(x.shape[3]):
                    y = decoder(x[:, :, i, j].unsqueeze(2).unsqueeze(3))

                    batch_loss += loss_func(y, padding_image[:, :, (i*32):(212+(i*32)), (j*32):(212+j*32)])

                    if((batch_index + 1) % 30 == 0):
                        image = make_grid(y, normalize=True)
                        save_image(image, f'result_images/{epoch}/{batch_index+1}_{i}_{j}.png')

            batch_loss.backward()
            decoder_optim.step()

            total_loss += batch_loss.item()
            # progress bar
            current_progress = (batch_index + 1) / len(train_loader) * 100
            progress_bar = '=' * int((batch_index + 1) * (20 / len(train_loader)))

            print(f'\r{" " * last_length}', end='')

            message = f'Epochs: {(epoch + 1):>{epoch_length}} / {args.epochs}, [{progress_bar:<20}] {current_progress:>6.2f}%, '
            message += f'loss: {batch_loss.item():.3f}'
            last_length = len(message) + 1

            print(f'\r{message}', end='')

        torch.save(decoder, f'checkpoint/decoder_{epoch}.pt')

        print(f'\r{" " * last_length}', end='')
        print(f'\rEpochs: {(epoch + 1):>{epoch_length}} / {args.epochs}, [{"=" * 20}], loss: {(total_loss/len(train_loader)):.3f}')
        scheduler.step()
