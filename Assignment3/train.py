from pickletools import optimize
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn
from network import Network, SmallNet
from torch.utils.data import random_split
from os.path import exists
import matplotlib.pyplot as plt

train_size = 45000
val_size = 5000
batch_size = 64
lr = 0.01
momentum = 0.9
epochs = 8

transform = transforms.Compose([transforms.ToTensor()])

data_set = torchvision.datasets.CIFAR10(root='./cifar-10/', train=True, download=True, transform=transform)
train_set, val_set = random_split(data_set, [train_size, val_size])
test_set = torchvision.datasets.CIFAR10(root='./cifar-10/', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)



model = Network(10)

# model = SmallNet(10)
# if exists("model_state_dict.pth"):
#     checkpoint = torch.load("model_state_dict.pth")
#     model.load_state_dict(checkpoint)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


n_train = len(train_set)
n_val = len(val_set)
# def train_step():
#     model.train()
#     for batch_idx, (inputs, label) in enumerate(train_loader, 0):
#         optimizer.zero_grad()

# loop over the epochs

epochs_loss_train = []
epochs_loss_val = []
epochs_acc_train = []
epochs_acc_val = []

for epoch in range(epochs):  

    # training
    model.train()
    train_loss = 0
    train_acc = 0
    total = 0
    iter = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
        
        # setting the parameters to zero
        optimizer.zero_grad()

        # forward step
        outputs = model(inputs)

        # getting the loss
        loss = criterion(outputs, labels)

        # backward step
        loss.backward()

        # optimizer step
        optimizer.step()

        train_loss += loss.item()
        _, predictions = outputs.max(1)
        train_acc += (predictions == labels).sum()

        iter += 1
        total += predictions.size(0)

        if batch_idx % 10 == 0:
            print(f"[{epoch+1} {batch_idx}] Train Loss: {loss.item()} Accuracy: {train_acc/total}")

    epochs_loss_train.append(train_loss/iter)
    epochs_acc_train.append(train_acc/total)
    

    
    # validation
    model.eval()
    val_loss = 0.0
    val_acc = 0
    iter = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(val_loader, 0):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predictions = outputs.max(1)
        val_acc += (predictions == labels).sum()
        iter += 1
        total += predictions.size(0)
        if batch_idx % 10 == 0:
            print(f"[{epoch+1} {batch_idx}] Test Loss: {loss.item()} Accuracy: {val_acc/total}")

    epochs_loss_val.append(val_loss/iter)
    epochs_acc_val.append(val_acc/total)

plt.plot(epochs_loss_train)
plt.plot(epochs_loss_val)
plt.show()
plt.plot(epochs_acc_train)
plt.plot(epochs_acc_val)
plt.show()
    
torch.save(model.state_dict(), 'model_state_dict.pth')