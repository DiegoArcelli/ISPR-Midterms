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
from torch.nn.functional import softmax

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def approx(x, digits):
    t = 10**digits
    return int(x*t)/t

def plot_attack(original : Tensor, noise : Tensor, attaccked, org : tuple, att : tuple, eps : float):
    org_class, org_prob = org
    att_class, att_prob = att
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(original.permute(1,2,0))
    # axs[0].set_title(txt, fontdict=None, loc='center',)
    axs[1].imshow(noise.permute(1,2,0))
    axs[2].imshow(attaccked.permute(1,2,0))

    for i in range(3):
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    org_title = r"$x$"
    noise_title = r'$sign(\nabla_x \ J(x;\theta))$'
    att_title = r'$x + \epsilon \cdot sign(\nabla_x \ J(x;\theta))$'
    axs[0].set_title(org_title, loc='center')
    axs[1].set_title(noise_title, loc='center')
    axs[2].set_title(att_title, loc='center')


    org_label = f"{org_class} with probability {approx(org_prob[0], 3)}%"
    noise_label = r"$\epsilon = $" + str(eps) 
    att_label = f"{att_class} with probability {approx(att_prob[0], 3)}%"
    axs[0].set_xlabel(org_label, loc='center')
    axs[1].set_xlabel(noise_label, loc='center')
    axs[2].set_xlabel(att_label, loc='center')

    print((original - attaccked).norm())


    plt.show()

def fast_gradient_sign(model : nn.Module, x : Tensor, eps, labels : Tensor, loss) -> Tensor:
    x.requires_grad = True
    model.zero_grad()
    out_org = model(x)
    org_prob, org_idx  = softmax(out_org, 1).max(1)
    org_class = class_names[int(org_idx)]

    l = loss(out_org, labels)
    l.backward()
    noise =  x.grad.sign()
    x_att = x + eps*noise
    x_att = torch.clamp(x_att, 0, 1)
    

    out_att = model(x_att)
    att_prob, att_idx = softmax(out_att, 1).max(1)
    att_class = class_names[int(att_idx)]

    x_att = x_att.detach()
    x = x.detach()
    for idx in range(len(x)):
        plot_attack(x[idx], noise[idx], x_att[idx], (org_class, org_prob), (att_class, att_prob), eps)
    return x_att


transform = transforms.Compose([transforms.ToTensor()])

test_set = torchvision.datasets.CIFAR10(root='./cifar-10/', train=False, download=True, transform=transform)

eps = 0.01
# model = SmallNet(10)
model = Network(10)
if exists("model_state_dict.pth"):
    checkpoint = torch.load("model_state_dict.pth")
    model.load_state_dict(checkpoint)




# image, label = test_set[1]
image, label = test_set[7]

image = image.reshape(1, 3, 32, 32)
label = torch.as_tensor([label])
model.eval()

att = fast_gradient_sign(model, image, eps, label, nn.CrossEntropyLoss())
