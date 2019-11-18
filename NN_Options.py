import torch.nn as nn
import torch.nn.functional as F

class ClassificationNet(nn.Module):
    def __init__(self, classes = 2):
        super(ClassificationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, classes)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x = F.relu(self.dropout(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.dropout(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)

class RegNet(nn.Module):
    def __init__(self, input_size):
        super(RegNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x