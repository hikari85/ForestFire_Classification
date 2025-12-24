from google.colab import drive
drive.mount('/content/drive')
fire_dataset_path = '/content/drive/MyDrive/Forest Fire Project/Datasets/FIRE Dataset'
wildfire_dataset_path = '/content/drive/MyDrive/Forest Fire Project/Datasets/Wildfire Detection Image Data'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import onnx
# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
])

# Load FIRE Dataset
fire_train = datasets.ImageFolder(root=f"{fire_dataset_path}/train", transform=transform)
fire_val = datasets.ImageFolder(root=f"{fire_dataset_path}/val", transform=transform)
fire_test = datasets.ImageFolder(root=f"{fire_dataset_path}/test", transform=transform)

# Load Wildfire Dataset
wildfire_train = datasets.ImageFolder(root=f"{wildfire_dataset_path}/train", transform=transform)
wildfire_val = datasets.ImageFolder(root=f"{wildfire_dataset_path}/val", transform=transform)
wildfire_test = datasets.ImageFolder(root=f"{wildfire_dataset_path}/test", transform=transform)

# Create DataLoaders
batch_size = 32

fire_train_loader = DataLoader(fire_train, batch_size=batch_size, shuffle=True)
fire_val_loader = DataLoader(fire_val, batch_size=batch_size, shuffle=False)
fire_test_loader = DataLoader(fire_test, batch_size=batch_size, shuffle=False)

wildfire_train_loader = DataLoader(wildfire_train, batch_size=batch_size, shuffle=True)
wildfire_val_loader = DataLoader(wildfire_val, batch_size=batch_size, shuffle=False)
wildfire_test_loader = DataLoader(wildfire_test, batch_size=batch_size, shuffle=False)

## Custom 3 layer CNN model
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification (fire or not)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))  # [B, 64, 28, 28]
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN().to(device)

summary(model, (3, 224, 224))  # Shows parameters, shapes

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

    print("Training complete.")
    return model


def evaluate_model(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100 * correct / total

model = CustomCNN()
trained_model = train_model(model, fire_train_loader, fire_val_loader, epochs=10, lr=0.001)
test_accuracy = evaluate_model(trained_model, fire_test_loader)
print(f"🔥 Test Accuracy on FIRE Dataset: {test_accuracy:.2f}%")
!pip install ptflops
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
    macs, params = get_model_complexity_info(CustomCNN(), (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print(f'FLOPs: {macs}')
    print(f'Params: {params}')

import torch

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
model = trained_model.to(device).eval()
dummy_input = torch.randn(1, 3, 224, 224).to(device)
# Warm-up
for _ in range(10):
    _ = model(dummy_input)

# Measure
import time
start = time.time()
for _ in range(100):
    _ = model(dummy_input)
end = time.time()

avg_time_ms = (end - start) / 100 * 1000
print(f"⏱️ Avg inference time per image: {avg_time_ms:.2f} ms")
Export CNN model
# Assuming 'model' is your trained model
torch.save(model.state_dict(), "custom_cnn_fire.pth")
!pip install onnx
import torch

# Make sure model is on CPU for export or on the same device as dummy input
model = CustomCNN()
model.load_state_dict(torch.load("custom_cnn_fire.pth"))
model.eval()

# Dummy input (batch size 1, 3 color channels, 224x224 image)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "custom_cnn_fire.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11
)

print("✅ Exported as custom_cnn_fire.onnx — ready for Netron!")
## ResNet-18 Model
Colab crashed re adding dependencies
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)  # Adjusting the final layer for 2 classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
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
    return model

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

trained_model = train_model(model, fire_train_loader, fire_val_loader, epochs=10, lr=0.001)
torch.save(trained_model.state_dict(), 'resnet18_fire_trained.pth')
!pip install torchsummary
from torchsummary import summary

# Print model summary
summary(model, (3, 224, 224))  # Assuming the input size is 224x224
!pip install fvcore
import torch
import time
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Ensure the model is on the correct device
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Initialize a dummy input to calculate FLOPs
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Compute FLOPs using fvcore
flops = FlopCountAnalysis(model, dummy_input)
print(f"FLOPs: {flops.total()}")

# Measure runtime on a batch of images
def measure_inference_time(model, data_loader):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            break  # Process just one batch
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / len(data_loader.dataset)
    return avg_inference_time

# Assuming you have a DataLoader for your validation set
batch_size = 32
val_dataset = datasets.ImageFolder(root=wildfire_dataset_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Measure the inference time per image
avg_inference_time = measure_inference_time(model, val_loader)
print(f"Average inference time per image: {avg_inference_time:.4f} seconds")

## MobileNetV2 Model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torchsummary import summary

# Load the pretrained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)

# Adjust the final fully connected layer for binary classification (2 classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# Move model to the available device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
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
    return model

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
trained_model_mobilenet = train_model(model, fire_train_loader, fire_val_loader, epochs=10, lr=0.001)

# Save the trained model
torch.save(trained_model_mobilenet.state_dict(), 'mobilenetv2_fire_trained.pth')

Low accuracy due to imbalanced data. changing class weights and modifying code
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Calculate class weights
num_fire = 630
num_non_fire = 164
total_samples = num_fire + num_non_fire

# Weight for each class (higher weight for the minority class "non-fire")
fire_weight = total_samples / (2 * num_fire)  # For "fire" class
non_fire_weight = total_samples / (2 * num_non_fire)  # For "non-fire" class

# Assign weights to the classes
class_weights = torch.tensor([fire_weight, non_fire_weight], device=device)

# Define model, loss function with class weights, and optimizer
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Adjust for 2 classes
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model as usual
trained_model = train_model(model, fire_train_loader, fire_val_loader, epochs=10, lr=0.001)

from torchsummary import summary

# Get summary of the trained model (assuming input size is 224x224 for MobileNetV2)
summary(trained_model, (3, 224, 224))
from fvcore.nn import FlopCountAnalysis
import time
import torch
import torchvision.models as models

# Move model to eval mode and CPU for FLOPs calculation
trained_model.eval()
trained_model.cpu()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# FLOPs calculation
flops = FlopCountAnalysis(trained_model, dummy_input)
print(f"FLOPs: {flops.total()}")

# Inference time calculation
trained_model.eval()
trained_model.to(device)
dummy_input = dummy_input.to(device)

with torch.no_grad():
    start_time = time.time()
    for _ in range(100):
        _ = trained_model(dummy_input)
    end_time = time.time()

avg_inference_time = (end_time - start_time) / 100
print(f"Average inference time per image: {avg_inference_time:.4f} seconds")
## EfficientNetB0
from torchvision.models import efficientnet_b0

# Load pretrained EfficientNetB0
model = efficientnet_b0(weights="IMAGENET1K_V1")

# Modify final classifier layer for 2 classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model = model.to(device)

# Reuse class weights from earlier
weights = torch.tensor([1.0, 630/164], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

trained_model = train_model(model, fire_train_loader, fire_val_loader, epochs=10, lr=0.001)
torch.save(trained_model.state_dict(), 'efficientnetb0_fire_trained.pth')
from torchsummary import summary
summary(trained_model, (3, 224, 224))
from fvcore.nn import FlopCountAnalysis
import time

dummy_input = torch.randn(1, 3, 224, 224).to(device)

# FLOPs
flops = FlopCountAnalysis(model, dummy_input)
print(f"FLOPs: {flops.total()}")

# Inference time
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)
end_time = time.time()
print(f"Average inference time per image: {(end_time - start_time)/100:.4f} seconds")
