import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score

# Set device to CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Define data transformations (No need to convert grayscale → RGB)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match DenseNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
])

# Load dataset (Ensure correct folder structure)
train_dataset = datasets.ImageFolder(root="dataset_224/malevis_train_val_224x224/train", transform=transform)
val_dataset = datasets.ImageFolder(root="dataset_224/malevis_train_val_224x224/val", transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define number of classes (26 for Malevis dataset)
num_classes = 26

# Initialize DenseNet-121 with random weights (training from scratch)
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)  # Modify classifier for 26 classes
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


# Training loop
num_epochs = 14
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "Adam_densenet_trained.pth")
print("Training complete! Model saved as Adam_densenet_trained.pth")

# ------------------------ VALIDATION ------------------------
print("\nStarting validation...")

# Set model to evaluation mode
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Compute performance metrics
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
val_accuracy = 100 * (sum([1 for x, y in zip(y_pred, y_true) if x == y]) / len(y_true))

# Print validation results
print(f"Validation Accuracy: {val_accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


