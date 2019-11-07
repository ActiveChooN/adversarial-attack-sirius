from __future__ import print_function
import argparse
import os
import numpy as np
from torchvision.datasets import ImageFolder
import torch
import torchvision
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import ExampleNet


def make_data(dataset, **kwargs):
    if dataset == "MNIST":
        return make_data_mnist(**kwargs)


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


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    writer = SummaryWriter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalar('Loss/train', loss.item(), batch_idx)
        entry = {
                    'grad_' + name: np.linalg.norm(p.grad.data.numpy()) 
                for name, p in model.named_parameters() if 'weight' in name
            }
        min_weights = {
                    'min_weight_' + name: p.min() 
                for name, p in model.named_parameters() if 'weight' in name
            }
        max_weights = {
                'max_weight_' + name: p.max() 
            for name, p in model.named_parameters() if 'weight' in name
        }
        std_weights = {
                'std_weight_' + name: p.std() 
            for name, p in model.named_parameters() if 'weight' in name
        }
        for name in entry:
            writer.add_scalar(name, entry[name], batch_idx)
        for name in min_weights:
            writer.add_scalar(name, min_weights[name], batch_idx)
        for name in max_weights:
            writer.add_scalar(name, max_weights[name], batch_idx)
        for name in std_weights:
            writer.add_scalar(name, std_weights[name], batch_idx)
        #writer.add_image('Image', data[0], batch_idx)  # Tensor


def test(model, device, test_loader):
    writer = SummaryWriter()
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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    writer.add_hparams({'lr': args.lr, 'optimizer': args.optimizer}, {'loss': test_loss, 'accuracy': 100. * correct / len(test_loader.dataset)})


def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', choices=['example'], default='example',
                        help='model name')
    parser.add_argument('--dataset', choices=['MNIST'], default='MNIST',
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

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = vars(args)

    kwargs.update({'num_workers': 1, 'pin_memory': True}) if use_cuda else {}

    train_loader, test_loader = make_data(args, kwargs)
    model = make_model(args, device)
    optimizer = make_optimizer(model, optimizer=args.optimizer, lr=args.lr,
                               momentum=args.momentum)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_path))

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, args.log_interval)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), args.save_path)


if __name__ == '__main__':
    main()
