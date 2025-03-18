import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Define the same network structure as before
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        return x

# Load trained model
model = MyNetwork()
model.load_state_dict(torch.load("trained_mnist_model.pth"))
model.eval()

# Define transformations for the images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

# Directory containing handwritten digit images
image_folder = "Images/"  # Change this to your actual folder
images = sorted(os.listdir(image_folder))  # Ensure correct order

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.ravel()

for i, image_name in enumerate(images):
    img_path = os.path.join(image_folder, image_name)
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    
    # Display image and prediction
    axes[i].imshow(Image.open(img_path), cmap='gray')
    axes[i].set_title(f"Pred: {prediction}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()