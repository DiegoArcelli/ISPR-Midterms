from network import Network, SmallNet, AvgNet
from torch import Tensor
import torch
from torchsummary import summary

x = torch.randn((10, 3, 32, 32))
# model = AvgNet(10)
# model = SmallNet(10)
model = Network(10)
y = model(x)

print(x.shape)
print(y.shape)
# print(model)

summary(model)