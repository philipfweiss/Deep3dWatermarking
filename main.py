from utils import *
from model import Net
from encoder import Encoder
from decoder import Decoder
from adversary import Adversary
from pprint import pprint
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from pointCloudDataset import PointCloudDataset
## Follow code here:
## https://github.com/pytorch/examples/blob/master/mnist/main.py#L2


def main():
    # pprint(vars(datasets))

    args = getargs()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("using cuda: ", use_cuda)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    ## Download the mnist datasets
    print(args)
    train_loader = torch.utils.data.DataLoader(
        PointCloudDataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        PointCloudDataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    #model = Net().to(device)

    k = args.k
    encoder = Encoder(k).to(device)
    decoder = Decoder(k).to(device)
    adversary = Adversary(k).to(device)
    params = list(adversary.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    ## Visualize one batch of training data
    dataiter = iter(train_loader)
    print(dataiter)
    # imshow(utils.make_grid(images))

    runner = RunModel()
    for epoch in range(args.epochs):
        for i, (data, encoding) in enumerate(runner.train(args, encoder, decoder, adversary, device, train_loader, optimizer, epoch)):
            with torch.no_grad():
                pass
                # concat = torch.cat((data, encoding), 0)
                #imshow(utils.make_grid(data[0:40, :, :, :]), utils.make_grid(encoding[0:40, :, :, :]), epoch*10 + i)

    runner.visualize()

    if (args.save_model):
        torch.save(encoder.state_dict(),"encoder.pt")
        torch.save(decoder.state_dict(),"decoder.pt")
        torch.save(adversary.state_dict(),"adversary.pt")

if __name__ == "__main__":
    main()
