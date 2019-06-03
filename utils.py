import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from collections import *
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import OrderedDict
from printVoxels import draw_voxels

def bce_loss(input, target):
    max_val = (-input).clamp(min=0)
    return input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

class RunModel:
    def __init__(self):
        self.train_losses = []
        self.train_decoder_losses = []
        self.train_encoder_losses = []
        self.train_adversary_losses = []
        self.train_image_gradients = []
        self.bits_correct = []
        self.total_bits = []

    def visualize(self):
        plt.figure(1)
        plt.title('Training Loss')
        plt.plot(self.train_losses, 'r-', label='Total Loss')
        plt.plot(self.train_decoder_losses, 'b-', label='Decoder Loss')
        plt.plot(self.train_encoder_losses, 'g-', label='Encoder Loss')
        plt.plot(self.train_adversary_losses, 'y-', label='Adversary Loss')
        plt.plot(self.train_image_gradients, 'm-', label='Sum of Image Gradient')
        plt.plot(np.divide(self.bits_correct, self.total_bits).tolist(), 'c-', label='Accuracy')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.xlabel("Batch")
        plt.savefig('results/losses.pdf')

    def train(self, args, encoder, decoder, adversary, device, train_loader, optimizer, epoch):
        encoder.train()
        decoder.train()
        adversary.train()

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            N, C, D, W, H = data.shape

            messageTensor = createMessageTensor(N, args.k, D, W, H, device)
            if (device == "cuda"):
                messageTensor = messageTensor.cuda()
            desiredOutput = messageTensor[:, :, 0, 0, 0]

            mask = torch.ceil(data)
            #mask = data
            #output, encoding = model(data, messageTensor)
            encoder_output = encoder(data, messageTensor, mask)
            decoder_output = decoder(encoder_output)
            adversary_output_fake = adversary(encoder_output)
            adversary_output_real = adversary(data)

            N = adversary_output_fake.size()[0]
            true_labels = torch.ones(N).to(device)
            false_labels = torch.zeros(N).to(device)

            decoderpredictions = decoder_output.round()
            numCorrect = torch.sum(decoderpredictions == desiredOutput).item() / float(N)

            a, b, c, e, f =  1, 0.70*1e5, 0.2, 0.001, 0.001
            decoder_loss = a * torch.mean(bce_loss(decoder_output, desiredOutput)) #decoder loss
            # diff_term = (encoder_output - data).norm(2) / (1 * D * H * W )
            diff_term = (encoder_output - data).norm(3) / (1 * D * H * W)
            encoder_loss = c * torch.mean(bce_loss(adversary_output_fake, true_labels)) + b * diff_term #encoder loss
            adversary_loss = e * torch.mean(bce_loss(adversary_output_real, true_labels) + f * bce_loss(adversary_output_fake, false_labels))

            image_grad = 0#torch.sum(torch.abs(data.grad))

            loss = encoder_loss + decoder_loss + adversary_loss
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                if batch_idx % 50*args.log_interval == 0:
                    yield data, encoder_output


                print(desiredOutput[0, :], decoder_output[0, :], "Percent correct: %d" % (numCorrect / args.k) )
                self.train_decoder_losses.append(decoder_loss.item() / a)
                self.train_adversary_losses.append(adversary_loss.item() / f)
                self.train_encoder_losses.append(encoder_loss.item() / (b + c))
                self.train_losses.append(loss.item()) #(epoch * args.batch_size + batch_idx,
                self.train_image_gradients.append(image_grad)
                self.bits_correct.append(numCorrect)
                self.total_bits.append(args.k)
                self.visualize()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tEncoder L: {:.5f}  \tDecoder L: {:.5f} \tAdversary L: {:.5f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), encoder_loss.item(), decoder_loss.item(), adversary_loss.item()))


    def test(self, args, encoder, decoder, adversary, device, test_loader, epoch):
        encoder.eval()
        decoder.eval()
        adversary.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                encoder_output = encoder(data)
                decoder_output = decoder(encoder_output)
                adversary_output_true = adversary(data)
                adversary_output_false = adversary(encoder_output)
                test_loss += F.cross_entropy(decoder_output, target, reduction='sum').item() # sum up batch loss
                pred = decoder_output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                num_cor = pred.eq(target.view_as(pred)).sum().item()
                print(num_cor)
                correct += num_cor
        self.test_losses.append(test_loss)
        # print(epoch, " reee ", test_loss)


def createMessageTensor(batchsize, message_len, depth, width, height, device):
    message_tensor = torch.zeros(batchsize, message_len, depth, width, height)
    for b in range(batchsize):
        message = np.random.randint(2, size=message_len) # defaults to 10
        tiled_message = torch.tensor(
            np.swapaxes(
                np.broadcast_to(message, (height, depth, width, message_len)),
                0, 3
            )
        )
        message_tensor[b, :, :, :, :] = tiled_message

    return message_tensor.to(device)

def imshow(im1, im2, im3, im4, e, i):
    im1 = im1.cpu().detach().numpy()
    im2 = im2.cpu().detach().numpy()
    im3 = im3.cpu().detach().numpy()
    im4 = im4.cpu().detach().numpy()

    fig = plt.figure(2)
    ax = fig.add_subplot(2, 2, 1)
    draw_voxels(im1, ax)
    ax = fig.add_subplot(2, 2, 2)
    draw_voxels(im2, ax)
    ax = fig.add_subplot(2, 2, 3)
    draw_voxels(im3, ax)
    ax = fig.add_subplot(2, 2, 4)
    draw_voxels(im4, ax)

    plt.title('Examples')

    plt.savefig(f'images/x_my_fig_{e}_{i}.pdf')


def pw__expirement(data):
    front = data[0, 0, :, :, :].sum(1).detach().numpy()
    img = plt.imshow(front)
    plt.show()

    # fig = plt.figure(2)
    # ax = fig.add_subplot(2, 2, 1, projection='3d')
    # draw_voxels(data, ax)
    # plt.show()
    # print(data.shape)

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
    parser.add_argument('--k', type=int, default=10, metavar='LR',
                        help='Bits in secret message')
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
    parser.add_argument('--local', action='store_true',
                        help='For local dev')
    parser.add_argument('--load-encoder', type=str, default='', metavar='N',
                        help='load pretrained encoder')
    args = parser.parse_args()
    return args
