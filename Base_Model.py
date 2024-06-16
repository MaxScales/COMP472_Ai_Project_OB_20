import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import seaborn as sns
import csv

# Define the paths to your dataset relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "Train")
test_path = os.path.join(script_dir, "Test")

# Check if paths exist
if not os.path.isdir(train_path) or not os.path.isdir(test_path):
    raise FileNotFoundError(f"Check if the Train and Test directories exist at {script_dir}")

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_images_and_labels(folder):
    file_paths = []
    labels = []
    class_names = sorted(os.listdir(folder))
    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(folder, class_name)
        for filename in os.listdir(class_folder):
            file_paths.append(os.path.join(class_folder, filename))
            labels.append(class_index)
    return file_paths, labels, class_names

def plot_class_distribution(labels, class_names):
    class_counts = [labels.count(i) for i in range(len(class_names))]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, class_counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    for bar, count in zip(bars, class_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{count}', ha='center', va='bottom')
    plt.show()

# Define your deeper CNN architecture
class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)  # 4 classes for emotions
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 128 * 6 * 6)
        x = self.fc_layer(x)
        return x

# Data transformation with more aggressive data augmentation
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Random scaling
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),  # More aggressive color adjustments
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Random erasing
    transforms.Normalize((0.5,), (0.5,))
])

# Load and preprocess your data
train_files, train_labels, class_names = load_images_and_labels(train_path)
test_files, test_labels, _ = load_images_and_labels(test_path)

# Create datasets
train_dataset = CustomDataset(train_files, train_labels, transform=transform)
test_dataset = CustomDataset(test_files, test_labels, transform=transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
split_test, split_valid = torch.utils.data.random_split(test_dataset, [0.5,0.5], torch.default_generator)

# Initialize the model, loss function, and optimizer
model = DeeperCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler with StepLR
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training the model
num_epochs = 50
best_model_wts = None
best_loss = float('inf')

acc_list=[]

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}, Accuracy:{(correct/total)*100:.2f}')

    # Step the learning rate scheduler
    scheduler.step()

    # Save the model if it has the best loss so far
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = model.state_dict()

# Load the best model weights
model.load_state_dict(best_model_wts)

# Evaluate the model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Calculate performance metrics
accuracy = accuracy_score(all_labels, all_preds)
macro_precision = precision_score(all_labels, all_preds, average='macro')
macro_recall = recall_score(all_labels, all_preds, average='macro')
macro_f1 = f1_score(all_labels, all_preds, average='macro')
micro_precision = precision_score(all_labels, all_preds, average='micro')
micro_recall = recall_score(all_labels, all_preds, average='micro')
micro_f1 = f1_score(all_labels, all_preds, average='micro')

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Macro Precision: {macro_precision:.4f}')
print(f'Macro Recall: {macro_recall:.4f}')
print(f'Macro F1 Score: {macro_f1:.4f}')
print(f'Micro Precision: {micro_precision:.4f}')
print(f'Micro Recall: {micro_recall:.4f}')
print(f'Micro F1 Score: {micro_f1:.4f}')

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the best model
torch.save(model.state_dict(), 'Models/best_model.pth')
