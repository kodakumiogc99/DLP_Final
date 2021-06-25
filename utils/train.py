import argparse
from tqdm import tqdm
from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import VGG


def train(args, trainloader, testloader, net, optim, loss_func):
    for epoch in range(args.epochs):
        net.train()

        train_loss = 0
        train_acc = 0

        for img, label in tqdm(trainloader):
            img = img.to(args.device)
            label = label.to(args.device)

            optim.zero_grad()

            output = net(img)

            loss = loss_func(output, label)
            train_loss += loss.item()

            pred = output.max(dim=1)[1]

            train_correct = (pred == label).sum()
            train_acc += train_correct

            loss.backward()
            optim.step()

        print(f'Epoch {epoch + 1}, Loss: {(train_loss / len(trainloader)):.3f}, Accuracy: {(train_acc / len(trainloader.dataset)):.3f}')

        test(args, testloader, net, loss_func)

        if (epoch + 1) % 1 == 0:
            torch.save(net, f'checkpoint/{args.model}_{epoch + 1}.pt')


def test(args, testloader, net, loss_func):
    net.eval()

    test_loss = 0
    test_acc = 0

    for img, label in testloader:
        img = img.to(args.device)
        label = label.to(args.device)

        out = net(img)

        loss = loss_func(out, label)

        test_loss += loss.item()
        pred = out.max(dim=1)[1]
        test_correct = (pred == label).sum()

        test_acc += test_correct

    print(f'Test, Loss: {(test_loss / len(testloader)):.3f}, Accuracy: {(test_acc / len(testloader.dataset)):.3f}')


def train_concept(args: argparse.Namespace, net: VGG, train_loader: DataLoader, test_loader: DataLoader, criterion: Callable, num_concepts: int) -> None:
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

    latent_dim = 512
    v = nn.Linear(latent_dim, num_concepts)
    g = nn.Linear(num_concepts, latent_dim)

    v = v.to(args.device)
    g = g.to(args.device)

    v_optimizer = optim.Adam(v.parameters(), lr=1e-5)
    g_optimizer = optim.Adam(g.parameters(), lr=1e-5)

    epochs = 1
    epoch_length = len(str(epochs))

    for epoch in range(epochs):
        loss = 0.0
        accuracy = 0.0

        last_length = 0

        for i, (image, label) in enumerate(train_loader):
            image = image.to(args.device)
            label = label.to(args.device)

            latents = phi(image)
            origin_shape = latents.shape

            v_optimizer.zero_grad()
            g_optimizer.zero_grad()

            latents = latents.reshape(latents.shape[0], latents.shape[1], -1).transpose(1, 2)

            latents = v(latents)
            latents = g(latents)

            latents = latents.transpose(1, 2).reshape(*origin_shape)

            latents = torch.flatten(latents, start_dim=1)
            output = h(latents)

            temp_loss = criterion(output, label)
            loss += temp_loss.item()

            temp_loss.backward()
            v_optimizer.step()
            g_optimizer.step()

            _, y_hat = output.max(dim=1)
            temp_accuracy = torch.sum(y_hat == label) / label.shape[0]
            accuracy += temp_accuracy

            # progress bar
            current_progress = (i + 1) / len(train_loader) * 100
            progress_bar = '=' * int((i + 1) * (20 / len(train_loader)))

            print(f'\r{" " * last_length}', end='')

            message = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}, [{progress_bar:<20}] {current_progress:>6.2f}%, loss: {temp_loss.item():.3f}, accuracy: {temp_accuracy:.3f}'
            last_length = len(message) + 1

            print(f'\r{message}', end='')

        loss /= len(train_loader)
        accuracy /= len(train_loader)

        print(f'\r{" " * last_length}', end='')
        print(f'\rEpochs: {(epoch + 1):>{epoch_length}} / {epochs}, [{"=" * 20}], loss: {loss:.3f}, accuracy: {accuracy:.3f}')

    with torch.no_grad():
        phi.eval()
        h.eval()

        v.eval()
        g.eval()

        loss = 0.0
        accuracy = 0.0

        last_length = 0
        for i, (image, label) in enumerate(test_loader):
            image = image.to(args.device)
            label = label.to(args.device)

            latents = phi(image)
            origin_shape = latents.shape

            latents = latents.reshape(latents.shape[0], latents.shape[1], -1).transpose(1, 2)

            latents = v(latents)
            latents = g(latents)

            latents = latents.transpose(1, 2).reshape(*origin_shape)

            latents = torch.flatten(latents, start_dim=1)
            output = h(latents)

            temp_loss = criterion(output, label)
            loss += temp_loss.item()

            _, y_hat = output.max(dim=1)
            temp_accuracy = torch.sum(y_hat == label) / label.shape[0]
            accuracy += temp_accuracy

            # progress bar
            current_progress = (i + 1) / len(test_loader) * 100
            progress_bar = '=' * int((i + 1) * (20 / len(test_loader)))

            print(f'\r{" " * last_length}', end='')

            message = f'Test, [{progress_bar:<20}] {current_progress:>6.2f}%, loss: {temp_loss.item():.3f}, accuracy: {temp_accuracy:.3f}'
            last_length = len(message) + 1

            print(f'\r{message}', end='')

        loss /= len(test_loader)
        accuracy /= len(test_loader)

        print(f'\r{" " * last_length}', end='')
        print(f'\rTest, [{"=" * 20}], loss: {loss:.3f}, accuracy: {accuracy:.3f}')

    with torch.no_grad():
        # v.weight = nn.Parameter(v.weight / torch.norm(v.weight, dim=1).reshape(-1, 1))
        for i in range(v.weight.shape[0]):
            for j in range(v.weight.shape[0]):
                if i == j:
                    continue

                value = torch.dot(v.weight[i], v.weight[j])
                print(value)

                if value > 0.95:
                    print(i, j)

    torch.save(v, 'checkpoint/v.pt')
    torch.save(g, 'checkpoint/g.pt')
