import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import *
import matplotlib.pyplot as plt
import numpy as np

class RunModel:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []

    def visualize(self):
        plt.subplot(2,1,1)
        plt.title('Training Loss')
        plt.plot(self.train_losses, 'o')
        plt.xlabel("Iteration")

        plt.subplot(2,1,2)
        plt.title('Test Loss')
        plt.plot(self.test_losses, 'ro')
        plt.xlabel("Epoch")
        plt.show()

    def train(self, args, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # print(data.shape, target.shape, 'reeee')
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            num_cor = pred.eq(target.view_as(pred)).sum().item()
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                self.train_losses.append(loss.item()) #(epoch * args.batch_size + batch_idx,
                print("Num correct: %d / %d" % (num_cor, args.batch_size) )
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(self, args, model, device, test_loader, epoch):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                num_cor = pred.eq(target.view_as(pred)).sum().item()
                print(num_cor)
                correct += num_cor
        self.test_losses.append(test_loss)
        # print(epoch, " reee ", test_loss)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def getargs():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    return args
