#python3 "/Users/aditshah/Desktop/PRCV/Project 5/Test.py"



import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Define the neural network
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
        self.log_softmax = nn.LogSoftmax(dim=1)  # Log Softmax for NLL Loss

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        return x

# Load the trained model
model = MyNetwork()
model.load_state_dict(torch.load("trained_mnist_model.pth"))
model.eval()

# Load test data
df = pd.read_csv("MNIST_CSV/mnist_test.csv")
data = df.iloc[:, 1:].values.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
labels = df.iloc[:, 0].values

_, X_test, _, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)), batch_size=10, shuffle=False)

# Run inference on the first 10 images
data_iter = iter(test_loader)
images, labels = next(data_iter)
outputs = model(images)

# Print results
for i in range(10):
    output_values = outputs[i].detach().numpy()
    pred_index = np.argmax(output_values)
    correct_label = labels[i].item()
    print(f"Sample {i+1}: {np.round(output_values, 2)} | Predicted: {pred_index} | Actual: {correct_label}")

# Plot the first 9 images with predictions
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    if i >= 9:
        break
    ax.imshow(images[i].squeeze(), cmap='gray')
    ax.set_title(f"Pred: {np.argmax(outputs[i].detach().numpy())}")
    ax.axis('off')

plt.tight_layout()
plt.show()
