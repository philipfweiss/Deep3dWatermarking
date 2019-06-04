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
import random
## Follow code here:
## https://github.com/pytorch/examples/blob/master/mnist/main.py#L2


def main():
    args = getargs()
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("using cuda: ", use_cuda)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dset = PointCloudDataset(args.local)
    train_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    k = args.k

    encoder = Encoder(k).to(device)
    decoder = Decoder(k).to(device)
    adversary = Adversary(k).to(device)

    if not args.save_model_to: Error("Must have a model to save to!")
    if args.load_model_from:
        encoder.load_state_dict(torch.load("./models/"+args.load_model_from + "-encoder.pt"))
        decoder.load_state_dict(torch.load("./models/"+args.load_model_from + "-decoder.pt"))
        adversary.load_state_dict(torch.load("./models/"+args.load_model_from + "-adversary.pt"))

    else:
        params = list(adversary.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
        optimizer = optim.Adam(params, lr=args.lr)
        ## Visualize one batch of training data
        dataiter = iter(train_loader)

        runner = RunModel(args)
        for epoch in range(args.epochs):
            for i, (data, encoding) in enumerate(runner.train(args, encoder, decoder, adversary, device, train_loader, optimizer, epoch)):
                with torch.no_grad():
                    # concat = torch.cat((data, encoding), 0)
                    if args.local: continue
                    im1 = random.randint(0, args.batch_size - 1)
                    im2 = random.randint(0, args.batch_size - 1)
                    imshow(data[im1, 0, :, :, :], data[im2, 0, :, :, :], encoding[im1, 0, :, :, :], encoding[im2, 0, :, :, :], epoch, i)

            if (args.save_model_to):
                print("saving model")
                torch.save(encoder.state_dict(), "./models/"+args.save_model_to+"-encoder.pt")
                torch.save(decoder.state_dict(), "./models/"+args.save_model_to+"-decoder.pt")
                torch.save(adversary.state_dict(), "./models/"+args.save_model_to+"-adversary.pt")

        runner.visualize()

    runner.test(args, encoder, decoder, adversary, device, test_loader, epoch)





if __name__ == "__main__":
    main()
