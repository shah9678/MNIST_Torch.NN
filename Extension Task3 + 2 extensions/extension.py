import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys

sys.path.append('/Users/aditshah/Desktop/PRCV/Project 5')
from Task1.MNIST import MyNetwork  

class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

model = MyNetwork()
model.load_state_dict(torch.load("Task1/trained_mnist_model.pth"))
model.eval()

for param in model.parameters():
    param.requires_grad = False

# Extension 2: Modify output layer to accommodate more Greek letters
num_classes = 6  # Updated to include more than three letters
model.fc2 = nn.Linear(50, num_classes)

for param in model.fc2.parameters():
    param.requires_grad = True

print("Modified Network:\n", model)

training_set_path = "/Users/aditshah/Desktop/PRCV/Project 5/greek_train"

greek_train = DataLoader(
    datasets.ImageFolder(training_set_path,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             GreekTransform(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ])),
    batch_size=5,
    shuffle=True
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc2.parameters(), lr=0.001)

def train_greek_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "trained_greek_model.pth")
    print("Greek letter model saved.")

    plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid()
    plt.show()

train_greek_model(model, greek_train, criterion, optimizer, epochs=10)


# -------------------- Extension 1: Evaluate More Dimensions --------------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import numpy as np

def evaluate_greek_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

evaluate_greek_model(model, greek_train)

def plot_confusion_matrix(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

class_names = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']  # Updated for more letters
plot_confusion_matrix(model, greek_train, class_names)


# -------------------- Extension 2: Try More Greek Letters --------------------

# Update training dataset path to include more letters
extended_training_set_path = "/Users/aditshah/Desktop/greek_train"

greek_extended_train = DataLoader(
    datasets.ImageFolder(extended_training_set_path,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             GreekTransform(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ])),
    batch_size=5,
    shuffle=True
)

# Train the model with more Greek letters
train_greek_model(model, greek_extended_train, criterion, optimizer, epochs=10)

# Evaluate the updated model
evaluate_greek_model(model, greek_extended_train)
plot_confusion_matrix(model, greek_extended_train, class_names)