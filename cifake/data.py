import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import kaggle
from constants import Constants
import random

def download_and_extract_dataset():
    """
    Downloads the dataset from Kaggle. Randomly takes 20% of the images to create a new validation set from the training data set!
    """
    dataset_name = "birdy654/cifake-real-and-ai-generated-synthetic-images"
    download_path = "cifake_dataset"

    os.makedirs(download_path, exist_ok=True)

    print(f"Downloading dataset: {dataset_name}")
    kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)

    print("Dataset downloaded and extracted successfully.")

    # Create validation dataset
    train_path = os.path.join(download_path, "train")
    val_path = os.path.join(download_path, "validation")
    os.makedirs(val_path, exist_ok=True)

    # Get all subdirectories (classes) in the train folder
    classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]

    for class_name in classes:
        train_class_path = os.path.join(train_path, class_name)
        val_class_path = os.path.join(val_path, class_name)
        os.makedirs(val_class_path, exist_ok=True)

        # Get all images in the class folder
        images = [f for f in os.listdir(train_class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Randomly select 20% of images for validation
        num_val_images = int(len(images) * 0.2)
        val_images = random.sample(images, num_val_images)

        # Move selected images to validation folder
        for img in val_images:
            src = os.path.join(train_class_path, img)
            dst = os.path.join(val_class_path, img)
            os.rename(src, dst)

    print("Validation dataset created successfully.")

def get_dataloaders(dataset_dir = Constants.DATASET_DIR, img_height=Constants.IMG_HEIGHT, img_width=Constants.IMG_WIDTH, batch_size=Constants.BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(os.path.join(dataset_dir, "train"), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(dataset_dir, "validation"), transform=transform)
    test_ds = datasets.ImageFolder(os.path.join(dataset_dir, "test"), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    download_and_extract_dataset()
