from torch import nn
from torch import Tensor

class Network(nn.Module):

    def __init__(self, classes) -> None:
        super().__init__()
        self.classes = classes

        self.conv1 = nn.Conv2d(3, 32, (3,3))
        self.pool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 64, (4, 4))
        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.pool = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, self.classes)
        self.softmax = nn.Softmax(1)


    def forward(self, x : Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class SmallNet(nn.Module):

    def __init__(self, classes) -> None:
        super().__init__()
        self.classes = classes
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 10, (8, 8))
        self.pool = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(90, self.classes)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x
