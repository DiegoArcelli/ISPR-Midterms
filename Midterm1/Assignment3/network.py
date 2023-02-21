from numpy import block
from torch import nn
from torch import Tensor


class AvgNet(nn.Module):

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
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class Block(nn.Module):


    def __init__(self, in_channels, out_channels, downsample = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3,3), (1,1), (1,1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3,3), (1,1) , (1,1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        if downsample:
            self.conv_downsample = nn.Conv2d(in_channels, out_channels, (3,3), (1,1) , (1,1))
            self.bn_downsample = nn.BatchNorm2d(out_channels)

    
    def forward(self, x : Tensor) -> Tensor:

        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample: 
            residual = self.conv_downsample(residual)
            residual = self.bn_downsample(residual)

        x += residual

        x = self.relu(x)

        return x


class Layer(nn.Module):


    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.block_1 = Block(in_channels, out_channels, True)
        self.block_2 = Block(out_channels, out_channels)
        self.block_3 = Block(out_channels, out_channels)
        self.block_4 = Block(out_channels, out_channels)
        self.block_5 = Block(out_channels, out_channels)
        self.block_6 = Block(out_channels, out_channels)
        self.block_7= Block(out_channels, out_channels)
        self.block_8 = Block(out_channels, out_channels)
        self.pool = nn.MaxPool2d((2,2))        


    def forward(self, x : Tensor) -> Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)
        x = self.pool(x)
        return x


class Network(nn.Module):

    def __init__(self, classes) -> nn.Module:
        super().__init__()
        self.classes = classes
        self.layer_1 = Layer(3, 32)
        self.layer_2 = Layer(32, 64)
        self.layer_3 = Layer(64, 128)
        self.layer_4 = Layer(128, 256)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, self.classes)
        self.relu = nn.ReLU(True)
        

    def forward(self, x : Tensor) -> Tensor:
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x