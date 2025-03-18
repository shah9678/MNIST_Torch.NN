import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import sys

sys.path.append('/Users/aditshah/Desktop/PRCV/Project 5')
from Task1.MNIST import MyNetwork  # Import the network class

# Load the trained model
model = MyNetwork()
model.load_state_dict(torch.load("Task1/trained_mnist_model.pth"))
model.eval()  # Set to evaluation mode

# Load the first training example from CSV
train_df = pd.read_csv("MNIST_CSV/mnist_train.csv")  # Replace with actual path
first_example = train_df.iloc[0, 1:].values.reshape(28, 28).astype(np.uint8)  # Exclude label

# Extract conv1 weights
conv1_weights = model.conv1.weight.detach().cpu().numpy()  # (10, 1, 5, 5)

# Apply filters to the first training image
filtered_images = []
with torch.no_grad():
    for i in range(10):  # Loop through the 10 filters
        kernel = conv1_weights[i, 0]  # Extract the filter
        filtered_img = cv2.filter2D(first_example, -1, kernel)  # Apply convolution
        
        # Normalize image for better visualization
        filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)
        filtered_images.append(filtered_img.astype(np.uint8))

# Plot original image + 10 filtered images in a 5x4 grid
fig, axes = plt.subplots(5, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    if i == 0:
        ax.imshow(first_example, cmap="gray")  # Show original image
        ax.set_title("Original", fontsize=10)
    elif i <= 10:
        ax.imshow(filtered_images[i-1], cmap="gray")  # Show filtered images
        ax.set_title(f"Filter {i}", fontsize=10)
    else:
        ax.axis("off")  # Hide extra grid spaces

    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
