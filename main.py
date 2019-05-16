from utils import *
from model import Net

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils

## Follow code here:
## https://github.com/pytorch/examples/blob/master/mnist/main.py#L2


def main():
    args = getargs()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    ## Download the mnist datasets
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)

    ## Change to adam
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    ## Visualize one batch of training data
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # imshow(utils.make_grid(images))

    runner = RunModel()
    for epoch in range(args.epochs):
        for i, (data, encoding) in enumerate(runner.train(args, model, device, train_loader, optimizer, epoch)):
            with torch.no_grad():
                # concat = torch.cat((data, encoding), 0)
                imshow(utils.make_grid(data[0:40, :, :, :]), utils.make_grid(encoding[0:40, :, :, :]), epoch*10 + i)

    runner.visualize()

    if (args.save_model):
        torch.save(model.state_dict(),"proj.pt")

main()
