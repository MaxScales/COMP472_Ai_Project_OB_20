import os
import random
import matplotlib.pyplot as plt
from skimage import io, color

# Define the paths to your dataset
train_path = "/Users/zaheerquraishi/COMP472/Train"
test_path = "/Users/zaheerquraishi/COMP472/Test"

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = io.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def plot_class_distribution(folder):
    classes = os.listdir(folder)
    num_images_per_class = []
    for class_name in classes:
        class_path = os.path.join(folder, class_name)
        num_images = len(os.listdir(class_path))
        num_images_per_class.append(num_images)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(classes, num_images_per_class, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')

    # Annotate each bar with its corresponding number of images
    for bar, num_images in zip(bars, num_images_per_class):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height(),
                 f'{num_images}',
                 ha='center',
                 va='bottom')

    plt.show()


def plot_pixel_intensity_distribution(images, title):
    plt.figure(figsize=(10, 5))
    for img in images:
        plt.hist(img.ravel(), bins=256, alpha=0.5, label='Intensity')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_sample_images(images, title):
    num_images = len(images)
    rows = 5
    cols = 3
    plt.figure(figsize=(15, 15))
    plt.suptitle(title)
    for i in range(rows * cols):
        plt.subplot(rows, cols*2, 2*i + 1)
        plt.imshow(images[i], cmap='gray')  # Display grayscale images
        plt.axis('off')
        plt.subplot(rows, cols*2, 2*i + 2)
        plt.hist(images[i].ravel(), bins=256, color='skyblue', alpha=0.7)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
    plt.show()


def visualize_dataset(train_path, test_path):
    # Load sample images from each class in training set
    train_classes = os.listdir(train_path)
    sample_images_per_class = {}
    for class_name in train_classes:
        class_path = os.path.join(train_path, class_name)
        class_images = load_images_from_folder(class_path)
        sample_images_per_class[class_name] = random.sample(class_images, 15)

    # Plot class distribution for training set
    plot_class_distribution(train_path)

    # Plot pixel intensity distribution for each class in training set
    for class_name, images in sample_images_per_class.items():
        plot_pixel_intensity_distribution(images, f'Pixel Intensity Distribution - {class_name}')

    # Plot sample images with pixel intensity histograms for each class in training set
    for class_name, images in sample_images_per_class.items():
        plot_sample_images(images, f'Sample Images - {class_name}')

visualize_dataset(train_path, test_path)
