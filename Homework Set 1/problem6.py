"""
Project 6: Why Deep Networks Exist

Take a nonlinear dataset.

Example

Two Moons

or

XOR

Train

Model A

Linear

Model B

Linear
↓

ReLU
↓

Linear

Compare

Decision boundaries
Accuracy
Training curves

Explain

Why can one solve the problem while the other cannot? This directly reinforces the lecture's point that stacked linear layers are still linear, while adding ReLU enables learning nonlinear functions.
"""


"""
The key to understanding this is that no amount of linear layers can turn an already linear function into a nonlinear function. 
Because of this, the data boundaries in a classification context if using Binary Classification (Logistic Regression) is based on a line, which most data won't be perfectly fit on.
Rather, if we were to use a nonlinear function to establish our data boundary, we have a significantly higher change of finding a boundary that's more effective than just a linear boundary, which is why
using activation functions like ReLU significantly help us boost our accuracy.
"""


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42) # Generates synthetic non-linear 2d datasets of interleaving half circle shapes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # Turns it to where at i=1, a new dimension is inserted: goes from (800) -> (800, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


model_A = nn.Sequential(
    nn.Linear(2, 16),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
model_B = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

epochs = 500
lr = 0.05
criterion = nn.BCELoss()

def train_model(model):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    losses = []
    for epoch in range(epochs):
        model.train()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses

print("Training Model A\n")
losses_A = train_model(model_A)
print("Training Model B\n")
losses_B = train_model(model_B)


def evaluate(model):
    model.eval()
    with torch.no_grad():
        preds = (model(X_test_t) >= 0.5).float()
        acc = accuracy_score(y_test_t.numpy(), preds.numpy())
    return acc

acc_A = evaluate(model_A)
acc_B = evaluate(model_B)

print(f"\nModel A (Stacked Linear) Test Accuracy: {acc_A * 100:.2f}%")
print(f"Model B (Linear + ReLU)   Test Accuracy: {acc_B * 100:.2f}%")

# Plot training curves

fig, axes = plt.subplots(1, 3, figsize=(18,5))

axes[0].plot(losses_A, label="Model A", color = 'red')
axes[0].plot(losses_B, label="Model B", color = 'green')
axes[0].set_title("Training Loss Curve")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Helper function to plot decision boundaries
def plot_boundary(model, ax, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Z = model(grid_tensor).reshape(xx.shape).numpy()
        
    ax.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=20)
    ax.set_title(title)

# Plot 2 & 3: Decision Boundaries
plot_boundary(model_A, axes[1], f"Model A: Linear Only\n(Acc: {acc_A*100:.1f}%)")
plot_boundary(model_B, axes[2], f"Model B: Linear + ReLU\n(Acc: {acc_B*100:.1f}%)")

plt.tight_layout()
plt.show()