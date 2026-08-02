"""
Objective

You are given a real-world tabular dataset and must design, train, and improve a neural network using PyTorch.

Recommended datasets (pick one):

UCI Adult Income
Bank Marketing
Heart Disease
Telco Customer Churn

All are binary classification problems, which match the lecture well.

Part 1: Data Exploration

Load the dataset.

Investigate:

Number of samples
Number of features
Missing values
Class imbalance
Feature distributions

Preprocess the data:

Encode categorical variables
Normalize numerical features
Split into train/validation/test sets
Part 2: Build Your First Deep Network

Create the following model:

Input
↓

Linear
↓

ReLU
↓

Linear
↓

ReLU
↓

Linear
↓

Output (1 logit)

Requirements

Use torch.nn.Sequential
Use ReLU after each hidden layer.
Do not apply Sigmoid() inside the model. The lecture recommends outputting raw values and using BCEWithLogitsLoss for numerical stability.

Train using:

BCEWithLogitsLoss
SGD
Learning rate = 0.01
Batch size = 64

Record:

Training loss
Validation accuracy
Part 3: Become the ML Engineer

Run the following experiments.

Experiment 1

Replace ReLU with LeakyReLU.

Compare:

Final accuracy
Training loss
Training speed

Write 3-5 sentences explaining why LeakyReLU might help if ReLU neurons "die."

Experiment 2

Try three learning rates:

0.1
0.01
0.001

Plot all three learning curves.

Explain:

Which converged fastest?
Which became unstable?
Which would you choose?
Experiment 3

Change the batch size.

Test:

16
64
256

Compare:

Epoch time
Validation accuracy
Stability of the loss curve

Relate your observations to why SGD learning curves fluctuate and why mini-batches reduce gradient variance.

Experiment 4

Add momentum.

Compare:

momentum = 0
momentum = 0.9

Plot both learning curves.

Explain why momentum often converges faster.

"""

import pandas as pd
import numpy as np
import torch 
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
df = pd.read_csv("Homework Set 2/telco_customer_churn.csv")
# Convert blank strings to NaN
df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
# Yes/No -> 1/0
yes_no_cols = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn"
]

for col in yes_no_cols:
    df[col] = df[col].apply(lambda x: 1 if x == "Yes" else 0)

# Female/Male -> 0/1
df["gender"] = df["gender"].apply(lambda x: 1 if x == "Male" else 0)

# MultipleLines
df["MultipleLines"] = df["MultipleLines"].apply(
    lambda x: 2 if x == "No phone service" else (1 if x == "Yes" else 0)
)

# InternetService
df["InternetService"] = df["InternetService"].apply(
    lambda x: 0 if x == "DSL" else (1 if x == "Fiber optic" else 2)
)

# Columns with "No internet service"
internet_cols = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

for col in internet_cols:
    df[col] = df[col].apply(
        lambda x: 2 if x == "No internet service" else (1 if x == "Yes" else 0)
    )

# Contract
df["Contract"] = df["Contract"].apply(
    lambda x: 0 if x == "Month-to-month"
    else (1 if x == "One year" else 2)
)

# PaymentMethod
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}
df["PaymentMethod"] = df["PaymentMethod"].apply(lambda x: payment_map[x])

# Remove customerID since it's just an identifier
df = df.drop(columns=["customerID"])
# Drop rows with missing values
df = df.dropna()


X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=1024, shuffle=True)



# Create our model
model = nn.Sequential(
    nn.Linear(X_train_tensor.shape[1], 64),
    nn.LeakyReLU(),
    nn.Linear(64, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 1)
)

loss = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

model.train()

epochs = 2000
losses = []

for i in range(epochs):
    y_pred = model(X_train_tensor)
    loss_value = loss(y_pred, y_train_tensor)
    print("Epoch: ", i+1, " Loss: ", loss_value.item())
    losses.append(loss_value.item())
    # This is done to make sure that gradients are not accumulated across epochs. If we don't do this, the gradients will be summed up across epochs and the model will not learn properly.
    optimizer.zero_grad()

    # Backpropogate the loss and update the weights
    loss_value.backward()

    # Optimizer step is the step where the weights are updated based on the gradients calculated in the backward pass. This is where the model learns and improves its predictions.
    optimizer.step()


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted_classes = (outputs > 0.5).float()
        accuracy = (predicted_classes == y_test).float().mean()
        losses.append(loss(outputs, y_test).item())
        print("Test Accuracy: {:.4f}".format(accuracy.item()))

evaluate(model, X_test_tensor, y_test_tensor)

plt.figure(figsize=(10, 5))
plt.plot(losses, label='Test Loss')
plt.title('Test Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()