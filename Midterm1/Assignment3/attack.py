from torch import Tensor, from_numpy
from torch import nn
import torch
from data import load_data
import matplotlib.pylab as plt
from os.path import exists
from network import Network, SmallNet
import torchvision.transforms as transforms


def fast_gradient_sign(model : nn.Module, x : Tensor, eps, label : int, loss) -> Tensor:

    x.requires_grad = True
    model.zero_grad()
    out = model(x)
    l = loss(out, label)
    l.backward()
    noise =  eps*x.grad.sign()
    x_att = x + noise
    x_att = torch.clamp(x_att, 0, 1)
    x_att = x_att.detach()
    print(x_att.shape)
    print(x_att[0].shape)
    plt.imshow(x_att[0].permute(1, 2, 0))
    plt.show()
    plt.imshow(noise[0].permute(1, 2, 0))
    plt.show()
    
    return x_att


x_train, y_train, _, _ = load_data()
x = x_train[:1]
x = torch.from_numpy(x)
image = x_train[0]
label = y_train[:1]
label = torch.from_numpy(label)
image = torch.from_numpy(image)


plt.imshow(image.permute(1, 2, 0))
plt.show()

# model = SmallNet(10)
model = Network(10)
if exists("model_state_dict.pth"):
    checkpoint = torch.load("model_state_dict.pth")
    model.load_state_dict(checkpoint)

attacked = fast_gradient_sign(model, x/255, 0.1, label, nn.CrossEntropyLoss())