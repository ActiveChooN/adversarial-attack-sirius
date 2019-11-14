from einops.layers.torch import Rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from memcnn import ReversibleBlock


class FirstBottleNeck(nn.Module):
    def __init__(self, c):
        super(FirstBottleNeck, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels=c,out_channels=c, kernel_size=(1,1))
        self.conv3x3 = nn.Conv2d(in_channels=c,out_channels=c,kernel_size=(3,3),padding=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=(1,1))
        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1x1_1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv3x3(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv1x1_2(y)
        return y


class BottleNeck(nn.Module):
    def __init__(self, c):
        super(BottleNeck, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels=c,out_channels=c,kernel_size=(1,1))
        self.conv3x3 = nn.Conv2d(in_channels=c,out_channels=c,kernel_size=(3,3),padding=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=(1,1))
        self.bn1 = nn.BatchNorm2d(c)
        self.bn2 = nn.BatchNorm2d(c)
        self.bn3 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        y = self.bn1(x)
        y = self.relu(y)
        y = self.conv1x1_1(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv3x3(y)
        y = self.bn3(y)
        y = self.relu(y)
        y = self.conv1x1_2(y)
        return y


class IRevNet(nn.Module):

    def __init__(self):
        super(IRevNet, self).__init__()
        self.split = Rearrange("b c (w1 w2) (h1 h2) -> b (c w2 h2) w1 h1", w2=2, h2=2)
        self.init_block = ReversibleBlock(FirstBottleNeck(6),implementation_fwd=-1,implementation_bwd=-1,keep_input=True,keep_input_inverse=True)
        self.block1 = ReversibleBlock(BottleNeck(6),implementation_bwd=-1,implementation_fwd=-1,keep_input=True,keep_input_inverse=True)
        self.block2 = ReversibleBlock(BottleNeck(24),implementation_bwd=-1,implementation_fwd=-1,keep_input=True,keep_input_inverse=True)
        self.block3 = ReversibleBlock(BottleNeck(96),implementation_bwd=-1,implementation_fwd=-1,keep_input=True,keep_input_inverse=True)
        self.block4 = ReversibleBlock(BottleNeck(384),implementation_bwd=-1,implementation_fwd=-1,keep_input=True,keep_input_inverse=True)
        self.block5 = ReversibleBlock(BottleNeck(1536),implementation_bwd=-1,implementation_fwd=-1,keep_input=True,keep_input_inverse=True)
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1_transfer = nn.Linear(3062, 100)
        self.fc2_transfer = nn.Linear(100, 10)

    def forward(self,x):
        y = self.split(x)
        y = self.init_block(y)
        y = self.block1(y)
        y = self.split(y)
        y = self.block2(y)
        y = self.split(y)
        y = self.block3(y)
        y = self.split(y)
        y = self.block4(y)
        y = self.split(y)
        y = self.block5(y)
        y = self.flat(y)
        if self.training:
            y1 = y[:, :10]
            y2 = y[:, 10:]
            y2 = self.fc1_transfer(y2)
            y2 = self.relu(y2)
            y2 = self.fc2_transfer(y2)
            return torch.cat((y1, y2), 1)
        else:
            return y
