from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1, bias=True
        )  # 28x28x1 -> 28x28x12
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # devides input by 2 28x28x12 -> 14x14x12
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True
        )  # 14x14x12 -> 14x14x24
        self.fc1 = nn.Linear(7 * 7 * 24, 10)  # Fully connected layer

    def forward(self, x):
        x = self.conv1(x)  # 28x28x1 -> 28x18x12
        x = F.relu(x)  # 28 x 28 x 12
        x = self.pool(x)  # 28x18x12 -> 14x14x12
        x = self.conv2(x)  # 14x14x12 -> 14x14x24
        x = F.relu(x)  # 14x14x25
        x = self.pool(x)  # 14x14x24 -> 7x7x24
        x = x.view(-1, 7 * 7 * 24)  # flatten
        x = self.fc1(x)

        return x

    def summary(self):
        print(self)
        print(self.conv1)
        print(self.conv2)
        print(self.fc1)


class dropoutCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True
        )  # 28x28x1 -> 28x28x12
        self.conv1_bn = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # devides input by 2 28x28x12 -> 14x14x12
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True
        )  # 14x14x12 -> 14x14x24
        self.conv2_bn = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(7 * 7 * 128, 30)  # Fully connected layer
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(30, 10)  # Fully connected layer

    def forward(self, x):
        x = self.conv1(x)  # 28x28x1 -> 28x18x12
        x = F.relu(x)  # 28 x 28 x 12
        x = self.pool(x)  # 28x18x12 -> 14x14x12
        x = self.conv2(x)  # 14x14x12 -> 14x14x24
        x = F.relu(x)  # 14x14x25
        x = self.pool(x)  # 14x14x24 -> 7x7x24
        x = x.view(-1, 7 * 7 * 128)  # flatten
        # add dropout
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def summary(self):
        print(self)
        print(self.conv1)
        print(self.conv2)
        print(self.fc1)
