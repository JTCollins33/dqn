import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=2)
        self.lin1 = nn.Linear(in_features=450+2, out_features=226)
        self.lin2 = nn.Linear(in_features=226, out_features=4)
        self.soft = nn.Softmax(dim=1)

    def forward(self, input, location):
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(1, -1)
        x = torch.cat((x, location), 1)
        x = F.relu(self.lin1(x))
        return self.soft(self.lin2(x))