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
from config import get_args


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)



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
