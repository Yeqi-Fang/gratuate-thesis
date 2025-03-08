import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize
from sklearn.model_selection import train_test_split

# Step 1: Generate data (Shepp-Logan Phantom and Sinogram)
def generate_data(size=128, samples=500):
    """
    Generate a dataset of sinograms and their corresponding images.
    """
    images = []
    sinograms = []
    for _ in range(samples):
        phantom = shepp_logan_phantom()
        phantom = resize(phantom, (size, size), mode='reflect', anti_aliasing=True)
        angle_steps = np.linspace(0., 180., size, endpoint=False)
        sinogram = radon(phantom, theta=angle_steps, circle=True)
        
        images.append(phantom)
        sinograms.append(sinogram)
    return np.array(sinograms), np.array(images)

# Generate data
sinograms, images = generate_data()
sinograms = sinograms[..., np.newaxis]  # Add channel dimension
images = images[..., np.newaxis]  # Add channel dimension

# Step 2: Prepare the dataset
X_train, X_test, y_train, y_test = train_test_split(sinograms, images, test_size=0.2, random_state=42)

# Permute dimensions for PyTorch (channels first)
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2),  # [batch_size, channels, height, width]
    torch.tensor(y_train, dtype=torch.float32).permute(0, 3, 1, 2)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2),
    torch.tensor(y_test, dtype=torch.float32).permute(0, 3, 1, 2)
)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 3: Define an improved CNN for reconstruction
class ImprovedCNNReconstructor(nn.Module):
    def __init__(self):
        super(ImprovedCNNReconstructor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Step 4: Train the improved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNNReconstructor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for sinogram, image in train_loader:
        sinogram, image = sinogram.to(device), image.to(device)
        optimizer.zero_grad()
        outputs = model(sinogram)
        loss = criterion(outputs, image)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()  # Adjust the learning rate

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Step 5: Test and visualize the results
model.eval()
with torch.no_grad():
    for sinogram, image in test_loader:
        sinogram, image = sinogram.to(device), image.to(device)
        reconstructed = model(sinogram)
        break

# Reshape for visualization
original_image = image[0].cpu().numpy().squeeze()
reconstructed_image = reconstructed[0].cpu().numpy().squeeze()

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Improved Reconstructed Image")
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
