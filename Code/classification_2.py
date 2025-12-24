from google.colab import drive
drive.mount('/content/drive')
wildfire_dataset_path = '/content/drive/MyDrive/Forest Fire Project/Datasets/Wildfire Detection Image Data'
pip install onnx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import onnx
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define the transformations for training and validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for the model
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load the dataset
train_data = datasets.ImageFolder(root=f'{wildfire_dataset_path}/train', transform=transform)
val_data = datasets.ImageFolder(root=f'{wildfire_dataset_path}/val', transform=transform)
test_data = datasets.ImageFolder(root=f'{wildfire_dataset_path}/test', transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 classes: fire, nofire
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model initialization
model = CustomCNN().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function for training
def train_model(model, train_loader, val_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

    print("Training complete.")

# Function to evaluate model on validation set
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# Train the model
train_model(model, train_loader, val_loader, epochs=10)

# Evaluate the model on the test set
test_acc = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")
!pip install fvcore

# Check the parameters
from torchsummary import summary
summary(model, (3, 224, 224))  # Assuming the input size is 224x224

# For FLOPs and inference time, you can use the same approach as previously shown
import torch
from fvcore.nn import FlopCountAnalysis
import time

# Get FLOPs
flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).to(device))
print("FLOPs:", flops.total())

# Test Inference time
start_time = time.time()
model(torch.randn(1, 3, 224, 224).to(device))  # Run inference on a random image
inference_time = time.time() - start_time
print(f"Average inference time per image: {inference_time:.4f} seconds")

## ResNet 18
from torchvision.models import resnet18

# Load pretrained ResNet18
resnet_model = resnet18(pretrained=True)

# Modify the final layer for binary classification
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 2)

# Move to device
resnet_model = resnet_model.to(device)
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    resnet_model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    resnet_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%')

print("Training complete.")

from torchsummary import summary
summary(resnet_model, input_size=(3, 224, 224))
!pip install fvcore  # if not already installed

from fvcore.nn import FlopCountAnalysis, parameter_count_table
import time

# FLOPs calculation
dummy_input = torch.randn(1, 3, 224, 224).to(device)
flops = FlopCountAnalysis(resnet_model, dummy_input)
print("FLOPs:", flops.total())

# Inference time per image
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = resnet_model(dummy_input)
end_time = time.time()

avg_inference_time = (end_time - start_time) / 100
print(f"Average inference time per image: {avg_inference_time:.4f} seconds")

##MobileNet V2
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_acc_history.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    print("Training complete.")
    return model, (train_loss_history, val_acc_history)

import torchvision.models as models
import torch.nn as nn

# Load pretrained MobileNetV2 model
mobilenet = models.mobilenet_v2(pretrained=True)

# Freeze all layers (optional)
for param in mobilenet.features.parameters():
    param.requires_grad = False

# Modify the classifier to match the number of output classes (2: fire, nofire)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)

# Move to device
mobilenet = mobilenet.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mobilenet.parameters(), lr=0.001)

# Train the model
mobilenet, history = train_model(mobilenet, train_loader, val_loader, criterion, optimizer, 10)

from torchsummary import summary
summary(mobilenet, input_size=(3, 224, 224))

from fvcore.nn import FlopCountAnalysis

dummy_input = torch.randn(1, 3, 224, 224).to(device)
flops = FlopCountAnalysis(mobilenet, dummy_input)
print(f"FLOPs: {flops.total()}")  # Or use flops.pretty_print() for formatted

import time

mobilenet.eval()
start_time = time.time()

with torch.no_grad():
    for _ in range(100):
        _ = mobilenet(dummy_input)

end_time = time.time()
avg_inference_time = (end_time - start_time) / 100
print(f"Average inference time per image: {avg_inference_time:.4f} seconds")

## EfficentNetB0
from torchvision import models
import torch.optim as optim

# Load the EfficientNetB0 model
efficientnet = models.efficientnet_b0(pretrained=True)
efficientnet.classifier[1] = torch.nn.Linear(efficientnet.classifier[1].in_features, 2)  # Adjust for 2 output classes

# Send the model to the device
efficientnet = efficientnet.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(efficientnet.parameters(), lr=0.001)

# Train the model
efficientnet, history = train_model(efficientnet, train_loader, val_loader, criterion, optimizer, 10)

summary(efficientnet, (3, 224, 224))
import torch
import time
from fvcore.nn import FlopCountAnalysis

# Move model to device
efficientnet = efficientnet.to(device)

# Create a dummy input
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Calculate FLOPs
flops = FlopCountAnalysis(efficientnet, dummy_input)
print(f"FLOPs: {flops.total()}")

# Measure inference time
start_time = time.time()
with torch.no_grad():
    efficientnet(dummy_input)
end_time = time.time()

inference_time = end_time - start_time
print(f"Average inference time per image: {inference_time:.4f} seconds")
