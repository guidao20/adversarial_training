import argparse
import logging
import time
from attack_method import Attack_methods
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from mnist_net import mnist_net
from adversarial_training import Adversarial_Trainings

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='../mnist-data', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['none', 'pgd', 'fgsm'])
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--alpha', default=0.375, type=float)
    parser.add_argument('--attack-iters', default=40, type=int)
    parser.add_argument('--lr-max', default=5e-3, type=float)
    parser.add_argument('--lr-type', default='cyclic')
    parser.add_argument('--fname', default='mnist_model1', type=str)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mnist_train = datasets.MNIST("mnist-data", train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)

    model = mnist_net().cuda()
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    if args.lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2//5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_type == 'flat': 
        lr_schedule = lambda t: args.lr_max
    else:
        raise ValueError('Unknown lr_type')


    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    
    adversarial_training = Adversarial_Trainings(args.epochs, train_loader, model, opt, args.epsilon, args.alpha, 40, args.lr_max, lr_schedule, 1, args.fname, logger)
    adversarial_training.fast_free_training()
    
    
    


if __name__ == "__main__":
    main()
