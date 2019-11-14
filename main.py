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
from ignite.engine import Engine, Events, create_supervised_trainer, \
    create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
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


def make_loss(model):
    if model == "example":
        return F.nll_loss


def make_metrics(metrics_set):
    if metrics_set == "default":
        return {
            "accuracy": Accuracy(),
            "loss": Loss(F.nll_loss)
        }


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
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)
    return train_loader, test_loader


def main():
    # Training settingsя
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', choices=['example'], default='example',
                        help='model name')
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], 
                        default='MNIST', help='dataset')
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
    parser.add_argument('--metrics', choices=["default"], default="default",
                        help="Metrics set for the evaluator")
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

    kwargs.update({'pin_memory': True} if use_cuda else {'pin_memory': False})

    train_loader, test_loader = make_data(**kwargs)
    model = make_model(args.model, device)
    optimizer = make_optimizer(model, optimizer=args.optimizer, lr=args.lr,
                               momentum=args.momentum)
    loss_fn = make_loss(args.model)
    metrics = make_metrics(args.metrics)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_path))

    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=getattr(logging, args.log_level.upper(), None))
    logger = BaseLogger(log_interval=args.log_interval,
                        train_len=len(train_loader))

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        logger.log_train(engine)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(test_loader)
        logger.log_test(evaluator)

    trainer.run(train_loader, max_epochs=args.epochs)

    if args.save_model:
        torch.save(model.state_dict(), args.save_path)

    logger.log_hparams({'Learning rate': args.lr, 'Optimizer': args.optimizer})


if __name__ == '__main__':
    main()
