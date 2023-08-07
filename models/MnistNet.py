import torch
from torch import nn
import torch.nn.functional as F


class MnistNet(nn.Module):
    def __init__(self, class_num=10):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x