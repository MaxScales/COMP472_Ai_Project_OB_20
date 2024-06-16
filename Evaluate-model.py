import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import seaborn as sns

# Define the model architectures
class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
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

class Variant1(nn.Module):
    def __init__(self):
        super(Variant1, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3 * 3 * 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 4)  # 4 classes for emotions
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

class Variant2(nn.Module):
    def __init__(self):
        super(Variant2, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=4),  # Modified kernel size and padding
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

# Load images and labels
def load_images_and_labels(folder):
    file_paths = []
    labels = []
    class_names = sorted(os.listdir(folder))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        for filename in os.listdir(class_folder):
            file_paths.append(os.path.join(class_folder, filename))
            labels.append(class_to_idx[class_name])
    return file_paths, labels, class_names

# Evaluate the model
def evaluate_model(model_path, model_class, test_loader, class_names):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Main function to load data and run evaluation
if __name__ == '__main__':
    # Define the paths to your dataset relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(script_dir, "Test")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load test data
    test_files, test_labels, class_names = load_images_and_labels(test_path)
    print(f"Class names: {class_names}")
    test_dataset = CustomDataset(test_files, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Choose model and path
    models = {
        'main': ('Main_Model.pth', MainModel),
        'variant1': ('model_variation1.pth', Variant1),
        'variant2': ('model_variation2.pth', Variant2)
    }

    # Prompt user to select the model
    print("Select the model to evaluate:")
    print("1. Main Model")
    print("2. Variant 1")
    print("3. Variant 2")
    choice = input("Enter the number of the model you want to evaluate: ")

    if choice == '1':
        model_key = 'main'
    elif choice == '2':
        model_key = 'variant1'
    elif choice == '3':
        model_key = 'variant2'
    else:
        raise ValueError("Invalid choice. Please enter 1, 2, or 3.")

    model_path, model_class = models[model_key]

    # Evaluate the chosen model
    evaluate_model(model_path, model_class, test_loader, class_names)
