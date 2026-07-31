"""
Project 3: Handwritten Digit Recognition (Multiclass)

Dataset

MNIST

Goal

Predict the digit (0-9)

Model

Single Linear Layer

(No hidden layers yet.)

Train using

CrossEntropyLoss

Report

Accuracy
Confusion Matrix

Then answer

Why does this work surprisingly well?

Where does it fail?

"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target.astype(int)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_x = MinMaxScaler()

# Scale Training Data
X_scaled_train = scaler_x.fit_transform(x_train)

# Scale Testing Data
X_scaled_test = scaler_x.transform(x_test)

X_train_tensor = torch.tensor(X_scaled_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long) # Long so that we can uses CrossEntropyLoss which expects class labels as integers
X_test_tensor = torch.tensor(X_scaled_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)


# 2. Wrap into PyTorch DataLoaders
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=1000, shuffle=False)

# 3. Instantiate the Model & Optimizer Directly (No Custom Class)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(), # Non-linearity to help the model learn more complex patterns
    nn.Linear(128, 10) # 10 outputs. CrossEntropyLoss manages Softmax for us!
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training on device:", device)
for epoch in range(10):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

model.eval()

with torch.no_grad():
    all_preds = []
    all_labels = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        print(outputs.shape)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))      