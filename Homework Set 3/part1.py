"""
Part 1: Build a Deep Network (30 pts)

Train two fully connected networks.

Model A
Input
↓

Linear(784→256)
↓

ReLU
↓

Linear(256→10)
Model B
Input
↓

Linear(784→256)

↓

ReLU

↓

(repeat 8 hidden layers)

↓

Linear(256→10)

Use identical hyperparameters.

Train both for 15 epochs.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data = FashionMNIST(root='./data', train=True, download=True, transform=None)

X_train, X_val, y_train, y_val = train_test_split(data.data, data.targets, test_size=0.2, random_state=42)

data_loader = DataLoader(list(zip(X_train, y_train)), batch_size=64, shuffle=True)

model_a = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
model_b = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

def train_model(model, data_loader, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch in data_loader:
            X_batch, y_batch = batch
            X_batch = X_batch.view(X_batch.size(0), -1).float()  # Flatten the images
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss_value = loss(outputs, y_batch)
            loss_value.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss_value.item()}")
        losses.append(loss_value.item())
    return losses

losses_a = train_model(model_a, data_loader)
losses_b = train_model(model_b, data_loader)

plt.figure(figsize=(10, 5))
plt.plot(losses_a, label='Model A')
plt.plot(losses_b, label='Model B')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()