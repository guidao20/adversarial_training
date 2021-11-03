import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='mnist-data', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['none', 'pgd', 'fgsm'])
    parser.add_argument('--attack-iters', default=40, type=int)
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--alpha', default=0.375, type=float)
    parser.add_argument('--lr-max', default=5e-3, type=float)
    parser.add_argument('--lr-type', default='cyclic')
    parser.add_argument('--fname', default='mnist_model1', type=str)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()
