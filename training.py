# Set CUDA to synchronous mode for detailed error reporting
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Optionally upgrade torch and torchvision
!pip install torch torchvision --upgrade

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

# Set device (Colab T4 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define your data directory.
# Ensure your folder structure is:
# /content/data/real/   <-- 1000 real images
# /content/data/fake/   <-- 1000 fake images
data_dir = 'give your file path'

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
}

# Load dataset (expects two subfolders: 'real' and 'fake')
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
print("Total images:", len(full_dataset))
print("Classes found:", full_dataset.classes)

# Split dataset into training (80%) and validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = data_transforms['val']

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
print("Training samples:", dataset_sizes['train'])
print("Validation samples:", dataset_sizes['val'])

# Debug: Print a sample batch of labels
inputs, labels = next(iter(train_loader))
print("Sample batch labels:", labels)

# ---------- MODEL INITIALIZATION WITH DEBUGGING ----------

# Option 1: Using EfficientNet-B3
try:
    print("Loading pre-trained EfficientNet-B3...")
    model_ft = models.efficientnet_b3(pretrained=True)
    # First, move the base model to GPU
    model_ft = model_ft.to(device)
    print("Base EfficientNet-B3 successfully moved to", device)
except Exception as e:
    print("Error while moving EfficientNet-B3 to device:", e)
    # If error persists, you can try an alternative model:
    print("Falling back to ResNet-50...")
    model_ft = models.resnet50(pretrained=True)
    model_ft = model_ft.to(device)
    # Replace the final fully connected layer for 2 classes
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2).to(device)

# If EfficientNet-B3 worked, replace its classifier layer
if "efficientnet_b3" in str(type(model_ft)).lower():
    try:
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, 2).to(device)
        print("Replaced classifier layer for EfficientNet-B3 and moved to", device)
    except Exception as e:
        print("Error while replacing classifier layer:", e)

# ---------- END MODEL INITIALIZATION ----------

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            dataloader = train_loader if phase == 'train' else val_loader

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (dataset_sizes['train'] if phase=='train' else dataset_sizes['val'])
            epoch_acc = running_corrects.double() / (dataset_sizes['train'] if phase=='train' else dataset_sizes['val'])
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# Train the model
num_epochs = 20
model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=num_epochs)

# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds

true_labels, predictions = evaluate_model(model_ft, val_loader)
print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=full_dataset.classes))
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predictions))
