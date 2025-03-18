import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys

sys.path.append('/Users/aditshah/Desktop/PRCV/Project 5')
# Import the MNIST model from Task 1
from Task1.MNIST import MyNetwork  # Ensure this module is available

# Define the transformation for Greek letter dataset
class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)  # Convert to grayscale
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)  # Scale
        x = torchvision.transforms.functional.center_crop(x, (28, 28))  # Crop
        return torchvision.transforms.functional.invert(x)  # Invert intensity to match MNIST

# Load the pre-trained MNIST model
model = MyNetwork()
model.load_state_dict(torch.load("Task1/trained_mnist_model.pth"))  # Load trained weights
model.eval()  # Set to evaluation mode

# Freeze the entire network
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer with a new one for 3 classes
model.fc2 = nn.Linear(50, 3)  # 3 output nodes (alpha, beta, gamma)

# Ensure only the new layer is trainable
for param in model.fc2.parameters():
    param.requires_grad = True

# Print modified network
print("Modified Network:\n", model)

# Define dataset path
training_set_path = "/Users/aditshah/Desktop/PRCV/Project 5/greek_train"  # Update with the actual path

# Load the Greek dataset
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

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc2.parameters(), lr=0.001)

# Train the model for Greek letters and track loss
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
    
    # Plot the training loss
    plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid()
    plt.show()

# Train and evaluate
train_greek_model(model, greek_train, criterion, optimizer, epochs=10)
