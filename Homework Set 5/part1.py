"""
Dataset

Use Fashion-MNIST (or MNIST).

Part 1: Autoencoder vs. Variational Autoencoder (60 pts)

Implement and train the following models:

Model A: Autoencoder
Input
↓

Encoder

↓

Latent Vector (32 dimensions)

↓

Decoder

↓

Reconstructed Image

Use MSE Loss.

Model B: Variational Autoencoder

Modify the encoder so it outputs:

Mean (μ)
Log-variance (logσ
2
)

Use the reparameterization trick to sample the latent vector, then train with:

Reconstruction Loss
KL Divergence Loss
Compare

For both models:

Plot the training loss.
Display 10 original images alongside their reconstructions.
Report the final reconstruction loss.

Answer:

Which model reconstructs images better?
Why does the VAE generally produce blurrier reconstructions than the standard autoencoder?
Part 2: Explore the Latent Space (25 pts)

Encode 1,000 test images into the latent space.

Use PCA or t-SNE to reduce the latent vectors to two dimensions.

Create a scatter plot where each point is colored by its class label.

Answer:

Do similar classes cluster together?
Which model produced a more structured latent space?
Why is having a structured latent space useful for generation?

"""


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Load Data
fashion_mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    def vae_loss(self, reconstructed, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x, reduction='sum')
        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
train_losses = [] # Track loss to plot later

model.train() # Set model to training mode
for epoch in range(epochs):
    epoch_loss = 0
    for batch_idx, (data, _) in enumerate(DataLoader(fashion_mnist_train, batch_size=64, shuffle=True)):
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        reconstructed, mu, logvar = model(data)
        loss = model.vae_loss(reconstructed, data, mu, logvar)
        
        # Backward pass now works because we are NOT in torch.no_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(DataLoader(fashion_mnist_train, batch_size=64))
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# 2. PLOT THE TRAINING LOSS
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title("VAE Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# 3. FIXED EVALUATION & VISUALIZATION
model.eval()
with torch.no_grad(): # no_grad() is correctly used here for evaluation
    # Grab exactly ONE batch of 10 images
    test_loader = DataLoader(fashion_mnist_test, batch_size=10, shuffle=True)
    data, _ = next(iter(test_loader))
    
    # Flatten and pass through model
    flat_data = data.view(data.size(0), -1)
    reconstructed, _, _ = model(flat_data)
    
    # Reshape back to 28x28 for plotting
    data = data.view(-1, 28, 28).numpy()
    reconstructed = reconstructed.view(-1, 28, 28).numpy()

# Plot original vs reconstructed
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    # Top row: Original images
    axes[0, i].imshow(data[i], cmap='gray')
    axes[0, i].set_title("Original")
    axes[0, i].axis('off')
    
    # Bottom row: Reconstructed images
    axes[1, i].imshow(reconstructed[i], cmap='gray')
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

# 1. EXTRACT LATENT VECTORS
model.eval()
latent_vectors = []
labels = []

# Process the entire test set to get a good visualization
test_loader = DataLoader(fashion_mnist_test, batch_size=512, shuffle=False)

with torch.no_grad():
    for data, target in test_loader:
        data = data.view(data.size(0), -1)
        # Pass data through only the encoder to get the 32D latent vectors
        latent = model.encoder(data) 
        
        latent_vectors.append(latent.numpy())
        labels.append(target.numpy())
from sklearn.manifold import TSNE
# Combine batches into single numpy arrays
latent_vectors = np.concatenate(latent_vectors, axis=0)
labels = np.concatenate(labels, axis=0)

print(f"Extracted latent vectors shape: {latent_vectors.shape}")

# 2. REDUCE DIMENSIONS (t-SNE)
print("Running t-SNE (this may take 15-30 seconds)...")
# Note: If you prefer PCA, swap this with: reducer = PCA(n_components=2)
reducer = TSNE(n_components=2, random_state=42)
latent_2d = reducer.fit_transform(latent_vectors)

# 3. PLOT THE LATENT SPACE
# Fashion MNIST class names
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 8))

# Create scatter plot, coloring by the target labels
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                      c=labels, cmap='tab10', alpha=0.7, s=10)

# Add a colorbar with the correct class names
cbar = plt.colorbar(scatter, ticks=range(10))
cbar.ax.set_yticklabels(classes)

plt.title("t-SNE Visualization of VAE Latent Space")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()