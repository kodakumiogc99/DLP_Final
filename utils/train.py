from tqdm import tqdm
import torch


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
            torch.save(net, f'checkpoint/vgg19_{epoch + 1}.pt')


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
