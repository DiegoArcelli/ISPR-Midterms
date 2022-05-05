from time import process_time_ns
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn
from network import Network, SmallNet
from torch.utils.data import random_split
from os.path import exists
from torch import Tensor
import matplotlib.pyplot as plt

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def plot_attack(original : Tensor, noise : Tensor, attaccked):
    fig, axs = plt.subplots(1, 3)
    txt="CuloCul\noCuloCuloCulo"
    axs[0].imshow(original.permute(1,2,0))
    axs[0].set_title(txt, fontdict=None, loc='center',)
    axs[1].imshow(noise.permute(1,2,0))
    axs[2].imshow(attaccked.permute(1,2,0))
    plt.show()

def fast_gradient_sign(model : nn.Module, x : Tensor, eps, labels : Tensor, loss) -> Tensor:
    x.requires_grad = True
    model.zero_grad()
    out = model(x)
    l = loss(out, labels)
    l.backward()
    noise =  x.grad.sign()
    x_att = x + eps*noise
    x_att = torch.clamp(x_att, 0, 1)
    # x_att = x_att.detach()
    # x = x.detach()
    # for idx in range(len(x)):
    #     plot_attack(x[idx], noise[idx], x_att[idx])
    return x_att



transform = transforms.Compose([transforms.ToTensor()])

test_set = torchvision.datasets.CIFAR10(root='./cifar-10/', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)

# model = SmallNet(10)
model = Network(10)
if exists("model_state_dict.pth"):
    checkpoint = torch.load("model_state_dict.pth")
    model.load_state_dict(checkpoint)

acc_org = 0
acc_att = 0
total = 0
eps = 0.01

model.eval()

for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    attacked = fast_gradient_sign(model, inputs, eps, labels, nn.CrossEntropyLoss())

    # forward + backward + optimize
    outputs_org = model(inputs)
    _, predictions_org = outputs_org.max(1)
    acc_org += (predictions_org == labels).sum()

    outputs_att = model(attacked)
    _, predictions_att = outputs_att.max(1)
    acc_att += (predictions_att == labels).sum()

    total += predictions_org.size(0)

    # print(outputs, labels)

print(f"Accuracy on original images {acc_org/total}")
print(f"Accuracy on attacked images {acc_att/total}")
