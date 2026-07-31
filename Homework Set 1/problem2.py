"""
Project 2: Predict Heart Disease (Binary Classification)

Dataset:

Heart Disease UCI Dataset

Goal:

Predict whether a patient has heart disease.

You'll practice:

Sigmoid
BCE Loss
Accuracy
Precision
Recall
Tasks

Train a binary classifier.

Evaluate

Accuracy
Precision
Recall
F1
ROC Curve
Questions

Would accuracy alone be enough?

Why or why not?



"""


import pandas
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from torch import nn, optim
df = pandas.read_csv("Homework Set 1/heart.csv")
df = df.dropna() # Removes empty rows

X_features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
y_feature = ["target"]

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scale Training Data

X_scaled_train = scaler_x.fit_transform(df_train[X_features])
y_scaled_train = scaler_y.fit_transform(df_train[y_feature])

# Scale Testing Data

X_scaled_test = scaler_x.fit_transform(df_test[X_features])
y_scaled_test = scaler_y.fit_transform(df_test[y_feature])

# Tensor Conversions:

X_train_tensor  = torch.tensor(X_scaled_train, dtype=torch.float32)
y_train_tensor  = torch.tensor(y_scaled_train, dtype=torch.float32)
X_test_tensor   = torch.tensor(X_scaled_test, dtype=torch.float32)
y_test_tensor   = torch.tensor(y_scaled_test, dtype=torch.float32)

# Move on to building the model, training, and evaluating it.

# 3. Define PyTorch Logistic Regression Model
# Binary Classification = Linear Layer + Sigmoid Activation
linear_layer = nn.Linear(in_features=len(X_features), out_features=1)
model = nn.Sequential(
    linear_layer,
    nn.Sigmoid()
)

criterion = nn.BCELoss() # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.5)

# 4. Training Loop with Autograd Inspection
epochs = 100

print("--- Training Loop ---")
for epoch in range(epochs):
    # Forward Pass
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    
    # Reset Gradients
    optimizer.zero_grad()
    
    # Backward Pass (PyTorch Autograd computes gradients here)
    loss.backward()
    
    # Inspect weight gradient L2 norm before the optimizer update step
    weight_grad_norm = linear_layer.weight.grad.norm().item()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Weight Grad Norm: {weight_grad_norm:.6f}")
        
    # Update Weights
    optimizer.step()

print("--- Training Complete ---")

model.eval()

with torch.no_grad():
    # Raw probabilities output by Sigmoid (range: 0.0 to 1.0)
    y_probs = model(X_test_tensor)
    
    # Threshold probabilities at 0.5 to get binary class predictions (0 or 1)
    y_preds = (y_probs >= 0.5).float()
    y_true_np = y_test_tensor.numpy()
    y_preds_np = y_preds.numpy()
    y_probs_np = y_probs.numpy()

    # 4. Calculate single-value metrics (using binary predictions)
    print("\n--- Test Set Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_true_np, y_preds_np):.4f}")
    print(f"Precision: {precision_score(y_true_np, y_preds_np):.4f}")
    print(f"Recall:    {recall_score(y_true_np, y_preds_np):.4f}")
    print(f"F1-Score:  {f1_score(y_true_np, y_preds_np):.4f}")

    # 5. ROC Metrics (use y_probs_np, NOT binary predictions)
    auc = roc_auc_score(y_true_np, y_probs_np)
    print(f"ROC AUC:   {auc:.4f}")

    # Extract points for plotting the ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true_np, y_probs_np)