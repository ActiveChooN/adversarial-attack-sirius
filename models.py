import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memcnn.models.revop import ReversibleBlock
from einops import rearrange


class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class RevNet(nn.Module):
    def __init__(self):
        super(RevNet, self).__init__()
        self.seq = nn.Sequential(ReversibleBlock(ArbBlock(channels=2),
                                                 keep_input=True,
                                                 keep_input_inverse=True),
                                 nn.Flatten())

    def forward(self, x):
        x = rearrange(x, "b c (w1 w2) (h1 h2)-> b (c w2 h2) w1 h1", w2=2, h2=2)
        x = self.seq(x)
        out = F.log_softmax(x[:,:10], dim=1)
        return out


class ArbBlock(nn.Module):
    def __init__(self, channels):
        super(ArbBlock, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels=channels,
                                           out_channels=channels,
                                           kernel_size=(3, 3), padding=1),
                                 nn.BatchNorm2d(num_features=channels),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        return self.seq(x)
