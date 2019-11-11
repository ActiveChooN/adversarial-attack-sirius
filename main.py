from __future__ import print_function
from utils.loggers import BaseLogger
import argparse
import os
import numpy as np
from torchvision.datasets import ImageFolder
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import ExampleNet
import logging


def make_data(dataset, **kwargs):
    if dataset == "MNIST":
        return make_data_mnist(**kwargs)
    if dataset == "CIFAR10":
        return make_data_cifar10(**kwargs)


def make_model(model, device):
    mdl = object()
    if model == "example":
        mdl = ExampleNet()
    return mdl.to(device)


def make_optimizer(model, optimizer, lr, momentum):
    if optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def make_data_mnist(train_batch_size, test_batch_size, num_workers,
                    pin_memory, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory)
    return train_loader, test_loader

def make_data_cifar10(train_batch_size, test_batch_size, num_workers,
                    pin_memory, **kwargs):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=num_workers,
                                              pin_memory=pin_memory)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=False, num_workers=num_workers,
                                             pin_memory=pin_memory)
    return train_loader, test_loader


def train(model, device, train_loader, optimizer, epoch, log_interval,
          logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        logger.log_train({"epoch": epoch, "loss": loss.item(),
                          "progress": 100. * batch_idx / len(train_loader)})
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     summary_writer.add_scalar('Loss/train', loss.item(), summary_step.get(1))
        # entry = {
        #             'grad_' + name: np.linalg.norm(p.grad.cpu().data.numpy())
        #             for name, p in model.named_parameters() if 'weight' in name
        #     }
        # min_weights = {
        #             'min_weight_' + name: p.min()
        #             for name, p in model.named_parameters() if 'weight' in name
        #     }
        # max_weights = {
        #         'max_weight_' + name: p.max()
        #         for name, p in model.named_parameters() if 'weight' in name
        # }
        # std_weights = {
        #         'std_weight_' + name: p.std()
        #         for name, p in model.named_parameters() if 'weight' in name
        # }
        # for name in entry:
        #     summary_writer.add_scalar(name, entry[name], summary_step.get())
        # for name in min_weights:
        #     summary_writer.add_scalar(name, min_weights[name], summary_step.get())
        # for name in max_weights:
        #     summary_writer.add_scalar(name, max_weights[name], summary_step.get())
        # for name in std_weights:
        #     summary_writer.add_scalar(name, std_weights[name], summary_step.get())
        # summary_writer.add_image('Image', data[0], summary_step.get()) #  Tensor


def test(model, device, test_loader, lr, optimizer, logger):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    logger.log_test({"loss": test_loss, "accuracy": accuracy})

    return test_loss, accuracy


def main():
    # Training settingsя
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', choices=['example'], default='example',
                        help='model name')
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='MNIST',
                        help='dataset')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', choices=['SGD'], default='SGD',
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging ' +
                        'training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-path', type=str, default='',
                        help='Path for Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For Loading the Model')
    parser.add_argument('--load-path', type=str, default='',
                        help='Path for Loading the Model')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='how many workers will be started (default: 1)')
    parser.add_argument('--log-level', choices=['NOTSET', 'DEBUG', 'INFO',
                        'WARNING', 'ERROR', 'CRITICAL'], default='INFO',
                        help='log level')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = vars(args)

    kwargs.update({'num_workers': 1, 'pin_memory': True}) if use_cuda else {}
    print(kwargs)
    logging.basicConfig(format='%(levelname)s: %(message)s',level=getattr(logging, args.log_level.upper(), None))
    logger = BaseLogger(log_interval=args.log_interval)

    train_loader, test_loader = make_data(**kwargs)
    model = make_model(args.model, device)
    optimizer = make_optimizer(model, optimizer=args.optimizer, lr=args.lr,
                               momentum=args.momentum)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_path))

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, args.log_interval,
              logger)
        test(model, device, test_loader, args.lr, args.optimizer, logger)
    if args.save_model:
        torch.save(model.state_dict(), args.save_path)

    logger.log_hparams({'Learning rate': args.lr, 'Optimizer': args.optimizer})


if __name__ == '__main__':
    main()
