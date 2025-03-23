import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Function to create Gabor filters
def create_gabor_kernels():
    kernels = []
    for theta in range(0, 180, 45):  # Different orientations
        kernel = cv2.getGaborKernel((5, 5), 1.0, np.deg2rad(theta), 2.0, 0.5, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)
    return np.array(kernels)

# Define the neural network with a fixed Gabor filter first layer
class MyNetwork(nn.Module):
    def __init__(self, gabor_kernels):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=len(gabor_kernels), kernel_size=5, bias=False)
        self.conv1.weight.data = torch.tensor(gabor_kernels).unsqueeze(1)
        self.conv1.weight.requires_grad = False  # Freeze first layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=len(gabor_kernels), out_channels=20, kernel_size=5)
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

# Training function
def train_network(train_loader, test_loader, model, criterion, optimizer, epochs=5):
    train_losses, test_losses = [], []
    num_examples_seen = []
    total_examples = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_examples += len(labels)
            running_loss += loss.item()

            train_losses.append(loss.item())
            num_examples_seen.append(total_examples)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append((total_examples, avg_test_loss))

        print(f"Epoch {epoch+1}/{epochs}: Test Loss: {avg_test_loss:.4f}")

    torch.save(model.state_dict(), "trained_mnist_gabor_model.pth")
    print("Model saved as trained_mnist_gabor_model.pth")

    # Plot loss graph
    plt.figure(figsize=(8, 6))
    plt.plot(num_examples_seen, train_losses, label="Train loss", color="blue")
    test_x, test_y = zip(*test_losses)
    plt.scatter(test_x, test_y, color="red", label="Test loss")
    plt.xlabel("Number of training examples seen")
    plt.ylabel("Negative Log-Likelihood Loss")
    plt.legend()
    plt.title("Training and Test Loss Curve")
    plt.show()

# Main function
def main(argv):
    df = pd.read_csv("MNIST_CSV/mnist_train.csv")
    data = df.iloc[:, 1:].values.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    labels = df.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)), batch_size=64, shuffle=False)

    gabor_kernels = create_gabor_kernels()
    model = MyNetwork(gabor_kernels)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    train_network(train_loader, test_loader, model, criterion, optimizer, epochs=5)

if __name__ == "__main__":
    main(sys.argv)