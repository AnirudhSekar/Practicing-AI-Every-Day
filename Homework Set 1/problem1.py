"""
Project 1: Predict California Housing Prices (Regression)

Dataset
California Housing Dataset

Goal:
Predict median house value.

You'll practice:

Linear Regression
MSE Loss
Gradient Descent
Tasks
Download the dataset.
Explore it.
Normalize features.
Build a Linear Regression model.
Train with PyTorch.
Plot training loss.
Report:
MSE
MAE
Sample predictions
Extension

Which features matter most?

Interpret the learned weights.

"""


from sklearn.metrics import mean_absolute_error
import torch
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df = pandas.read_csv("Homework Set 1/housing.csv")
df = df.dropna() # Removes empty rows

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

input_features = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
target_feature = ["median_house_value"]

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scale Training Data
X_train_scaled = scaler_X.fit_transform(df_train[input_features])
y_train_scaled = scaler_y.fit_transform(df_train[target_feature])

# Scale Testing Data
X_test_scaled = scaler_X.transform(df_test[input_features])
y_test_scaled = scaler_y.transform(df_test[target_feature])

# Convert back to dataframes/tensors cleanly
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


# Normalize the dataframe mathematically with this formula:
# df = (df - df.min()) / (df.max() - df.min()) 




# Create linear regression model with PyTorch

linear_network = torch.nn.Linear(X_train_scaled.shape[1], 1)
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(linear_network.parameters(), lr=0.01)

epochs = 100

for i in range(epochs):
    y_pred = linear_network(X_train_tensor)
    loss_value = loss(y_pred, y_train_tensor)
    print("Epoch: ", i+1, " Loss: ", loss_value.item())
    # This is done to make sure that gradients are not accumulated across epochs. If we don't do this, the gradients will be summed up across epochs and the model will not learn properly.
    optimizer.zero_grad()

    # Backpropogate the loss and update the weights
    loss_value.backward()

    # Optimizer step is the step where the weights are updated based on the gradients calculated in the backward pass. This is where the model learns and improves its predictions.
    optimizer.step()


# Turn model onto evaluation mode. This is done to make sure that the model is not learning anymore and is only making predictions.

linear_network.eval()

with torch.no_grad():
    test_data = torch.tensor(X_test_scaled, dtype=torch.float32)
    test_predictions = linear_network(test_data)
    
    loss_value = loss(test_predictions, torch.tensor(y_test_scaled, dtype=torch.float32).view(-1,1)) # .view(-1, 1) turns it from a 1xN tensor to an Nx1, from a single row to a single column which matches our predictions

    print("Average MSE on test set: ", loss_value.item())
    loss_value_mae = mean_absolute_error(scaler_y.inverse_transform(y_test_scaled), scaler_y.inverse_transform(test_predictions.numpy()))
    print("Average MAE on test set: ", loss_value_mae)


# Get weights of the linear model:

linear_weights = linear_network.weight.data.numpy()

print("Weights of the linear model: ", linear_weights)
