import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

# Define the neural network
class MyNetwork(nn.Module):
    def __init__(self, num_conv_layers=2, num_filters=20, dropout_rate=0.3):
        super(MyNetwork, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=5) if num_conv_layers > 1 else None
        self.conv3 = nn.Conv2d(in_channels=num_filters * 2, out_channels=num_filters * 3, kernel_size=5) if num_conv_layers > 2 else None
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(num_filters * 12 * 12 if num_conv_layers == 1 else num_filters * 2 * 4 * 4 if num_conv_layers == 2 else num_filters * 3 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        if self.num_conv_layers > 1:
            x = self.relu(self.pool(self.conv2(x)))
        if self.num_conv_layers > 2:
            x = self.relu(self.pool(self.conv3(x)))
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
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_examples += len(labels)
            running_loss += loss.item()

            train_losses.append(loss.item())
            num_examples_seen.append(total_examples)

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append((total_examples, avg_test_loss))
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}: Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    training_time = time.time() - start_time
    return accuracy, training_time

# Main function
def main(argv):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(root='/Users/aditshah/Desktop/PRCV/Project 5/MNIST_fashion', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='/Users/aditshah/Desktop/PRCV/Project 5/MNIST_fashion', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Define the configurations to explore
    num_conv_layers_options = [1, 2, 3]
    num_filters_options = [10, 20, 30]
    dropout_rate_options = [0.2, 0.3, 0.4]

    results = []

    for num_conv_layers in num_conv_layers_options:
        for num_filters in num_filters_options:
            for dropout_rate in dropout_rate_options:
                print(f"Training model with num_conv_layers={num_conv_layers}, num_filters={num_filters}, dropout_rate={dropout_rate}")
                model = MyNetwork(num_conv_layers=num_conv_layers, num_filters=num_filters, dropout_rate=dropout_rate)
                criterion = nn.NLLLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                accuracy, training_time = train_network(train_loader, test_loader, model, criterion, optimizer, epochs=5)
                results.append({
                    'num_conv_layers': num_conv_layers,
                    'num_filters': num_filters,
                    'dropout_rate': dropout_rate,
                    'accuracy': accuracy,
                    'training_time': training_time
                })

    # Save results to a DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiment_results.csv', index=False)
    print("Experiment results saved to experiment_results.csv")

if __name__ == "__main__":
    main(sys.argv)