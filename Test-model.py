import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the paths to your dataset relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.join(script_dir, "Test")

# Check if the test path exists
if not os.path.isdir(test_path):
    raise FileNotFoundError(f"Check if the Test directory exists at {script_dir}")


# Load the class names
def load_images_and_labels(folder):
    file_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(folder, class_name)
        for filename in os.listdir(class_folder):
            file_paths.append(os.path.join(class_folder, filename))
            labels.append(class_index)
    return file_paths, labels, class_names


_, _, class_names = load_images_and_labels(test_path)


# Define your CNN architectures (adjust these to match your saved model's architectures)
class LeakyReLuCNN(nn.Module):
    def __init__(self):
        super(LeakyReLuCNN, self).__init__()
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
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 128 * 6 * 6)
        x = self.fc_layer(x)
        return x


class ModelVariation1(nn.Module):
    def __init__(self):
        super(ModelVariation1, self).__init__()
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
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


class ModelVariation2(nn.Module):
    def __init__(self):
        super(ModelVariation2, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=4),
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
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 128 * 6 * 6)
        x = self.fc_layer(x)
        return x


# Load the saved model
def load_model(model_choice):
    if model_choice == 1:
        model = LeakyReLuCNN()
        model_name = 'Main_Model.pth'
    elif model_choice == 2:
        model = ModelVariation1()
        model_name = 'model_variation1.pth'
    elif model_choice == 3:
        model = ModelVariation2()
        model_name = 'model_variation2.pth'
    else:
        raise ValueError("Model choice not recognized. Choose 1, 2, or 3.")

    model.load_state_dict(torch.load(model_name))
    model.eval()
    return model


# Transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Predict and display the results
def predict_and_display(model, image_path, class_names):
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = class_names[predicted.item()]

    actual_label = os.path.basename(os.path.dirname(image_path))

    plt.imshow(image, cmap='gray')
    plt.title(f'Actual: {actual_label} | Predicted: {predicted_label}')
    plt.axis('off')
    plt.show()


# Select a random image from the test set
def select_random_image(test_path):
    all_images = []
    for root, _, files in os.walk(test_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))
    return random.choice(all_images)


# Select a specific image from the test set
def select_specific_image(test_path, emotion, image_name):
    return os.path.join(test_path, emotion, image_name)


# Main function
def main():
    print("Choose the model to use:")
    print("1: Main_Model.pth")
    print("2: model_variation1.pth")
    print("3: model_variation2.pth")
    model_choice = int(input("Enter the model number (1, 2, or 3): "))
    model = load_model(model_choice)

    choice = input("Do you want to select a random image or a specific image? (random/specific): ")
    if choice == 'random':
        image_path = select_random_image(test_path)
    elif choice == 'specific':
        emotion = input(f"Enter the emotion ({', '.join(class_names)}): ")
        image_name = input("Enter the image name (with extension, e.g., 'image1.png'): ")
        image_path = select_specific_image(test_path, emotion, image_name)
    else:
        raise ValueError("Choice not recognized. Choose 'random' or 'specific'.")

    predict_and_display(model, image_path, class_names)


if __name__ == '__main__':
    main()
