import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from constants import Constants

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

    print("Train dataset classes:", train_ds.classes)
    print("Validation dataset classes:", val_ds.classes)
    print("Test dataset classes:", test_ds.classes)

    return train_loader, val_loader, test_loader
