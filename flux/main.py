from model import FakeDetectorCNN
from data import get_dataloaders

if __name__ == "__main__":
    # train_loader, val_loader, test_loader = get_dataloaders()
    # fdcnn_noaug = FakeDetectorCNN(invert_and_saturate=False, name="NoAug")
    # fdcnn_noaug.train_model(train_loader, val_loader)
    # fdcnn_noaug.test_model(test_loader)

    train_loader, val_loader, test_loader = get_dataloaders()
    fdcnn_aug = FakeDetectorCNN(invert_and_saturate=True, name="Aug")
    fdcnn_aug.train_model(train_loader, val_loader)
    fdcnn_aug.test_model(test_loader)
