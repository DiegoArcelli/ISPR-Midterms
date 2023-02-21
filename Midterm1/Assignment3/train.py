from pickletools import optimize
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn
from network import Network, SmallNet, AvgNet
from torch.utils.data import random_split
from os.path import exists
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

train_size = 45000
val_size = 5000
batch_size = 128
lr = 0.01
momentum = 0.9
epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])

data_set = torchvision.datasets.CIFAR10(root='./cifar-10/', train=True, download=True, transform=transform)
train_set, val_set = random_split(data_set, [train_size, val_size])
test_set = torchvision.datasets.CIFAR10(root='./cifar-10/', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)



model = AvgNet(10)
model.to(device)
# model = SmallNet(10)
# model = SmallNet(10)

# model = SmallNet(10)
# if exists("model_state_dict.pth"):
#     checkpoint = torch.load("model_state_dict.pth")
#     model.load_state_dict(checkpoint)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


n_train = len(train_set)
n_val = len(val_set)

epochs_loss_train = []
epochs_loss_val = []
epochs_acc_train = []
epochs_acc_val = []

max_val_acc = 0

for epoch in range(epochs):  

    # training
    model.train()
    train_loss = 0
    train_acc = 0
    total = 0
    iter = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
        
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predictions = outputs.max(1)
        train_acc += float((predictions == labels).sum())

        iter += 1
        total += predictions.size(0)

        if batch_idx % 30 == 0:
            print(f"[{epoch+1} {batch_idx}] Train Loss: {train_loss/iter} Train Accuracy: {train_acc/total}")

    epochs_loss_train.append(train_loss/iter)
    epochs_acc_train.append(train_acc/total)
    

    
    # validation
    model.eval()
    val_loss = 0.0
    val_acc = 0
    iter = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(val_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predictions = outputs.max(1)
        val_acc += float((predictions == labels).sum())
        iter += 1
        total += predictions.size(0)
        if batch_idx % 3 == 0:
            print(f"[{epoch+1} {batch_idx}] Validation Loss: {val_loss/iter} Validation Accuracy: {val_acc/total}")

    epochs_loss_val.append(val_loss/iter)
    epochs_acc_val.append(val_acc/total)

torch.save(model.state_dict(), 'model_state_dict.pth')

# plotting the loss through the epochs
plt.plot(epochs_loss_train, label="Train loss")
plt.plot(epochs_loss_val, label="Val loss")
plt.legend()
plt.show()

# plotting the accuracy through the epochs
plt.plot(epochs_acc_train, label="Train accuracy")
plt.plot(epochs_acc_val, label="Val accuracy")
plt.legend()
plt.show()
    
