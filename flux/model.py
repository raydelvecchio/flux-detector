import torch
import torch.nn as nn
import torch.optim as optim
from constants import Constants
import torchvision.transforms.functional as TF
import time
from PIL import Image
import os
import numpy as np
import random

class FakeDetectorCNN(nn.Module):
    def __init__(self, invert_and_saturate: bool, name: str = "FakeDetectorCNN"):
        super(FakeDetectorCNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.invert_and_saturate = invert_and_saturate
        self.name = name
        self.training = False
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # calculate size of the flattened feature map using the image height and
        self.feature_size = self._get_conv_output_size((3, Constants.IMG_HEIGHT, Constants.IMG_WIDTH))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.criterion = nn.BCELoss()
        self.lr = 1e-3
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.saturation_factor = 8

        self.to(self.device)
        self.print_architecture()

    def _get_conv_output_size(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.conv(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def print_architecture(self):
        print(f"Architecture of {self.name}:")
        print("\nModel Summary:")
        print(self)
        print(f"\nTotal parameters: {sum(p.numel() for p in self.parameters())}")
        print(f"\nTrainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        print(f"\nInvert and Saturate: {self.invert_and_saturate}")
        print(f"Device: {self.device}")
        print(f"Loss Function: {self.criterion.__class__.__name__}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Flattened feature size: {self.feature_size}")

    def forward(self, x):
        original_x = x.clone()
        
        if self.invert_and_saturate:
            x = 1 - x  # invert the colors of the image
            x = TF.adjust_saturation(x, self.saturation_factor)  # maximize the saturation of the image

        if self.training and random.random() <= 0.02:  # 2% chance of saving a sample
            os.makedirs('training_samples', exist_ok=True)
            
            # get class prediction
            with torch.no_grad():
                class_pred = self.fc(self.conv(x[0].unsqueeze(0).to(self.device)))
            class_label = "FAKE" if class_pred.item() > 0.5 else "REAL"
            
            # save processed image (inverted and saturated if applicable)
            processed_img = x[0].cpu().detach().permute(1, 2, 0).numpy()
            processed_img = (processed_img * 255).astype(np.uint8)
            processed_img = Image.fromarray(processed_img)
            timestamp = time.time()
            processed_filename = f'training_samples/processed_{class_label}_sf{self.saturation_factor}_{timestamp}.png'
            processed_img.save(processed_filename)
            print(f"Saved processed image sample: {processed_filename}")
            
            # save the original
            orig_img = original_x[0].cpu().detach().permute(1, 2, 0).numpy()
            orig_img = (orig_img * 255).astype(np.uint8)
            orig_img = Image.fromarray(orig_img)
            orig_filename = f'training_samples/original_{class_label}_{timestamp}.png'
            orig_img.save(orig_filename)
            print(f"Saved original image sample: {orig_filename}")

        x = self.conv(x)
        x = self.fc(x)
        return x

    def train_model(self, train_loader, val_loader):
        print(f"Starting training for {self.name}...")
        self.training = True
        for epoch in range(Constants.EPOCHS):
            self.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device).float().unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += ((outputs > 0.5) == labels).sum().item()
                train_total += labels.size(0)

            # VALIDATION BEGINS HERE
            
            self.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device).float().unsqueeze(1)
                    outputs = self(images)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    val_correct += ((outputs > 0.5) == labels).sum().item()
                    val_total += labels.size(0)

            print(f"{self.name} - Epoch {epoch+1}/{Constants.EPOCHS}")
            print(f"{self.name} - Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_correct/train_total:.4f}")
            print(f"{self.name} - Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_correct/val_total:.4f}")

        self.training = False
        print(f"Training finished for {self.name}.")

    def test_model(self, test_loader):
        print(f"Starting testing for {self.name}...")
        self.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device).float().unsqueeze(1)
                outputs = self(images)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                test_correct += ((outputs > 0.5) == labels).sum().item()
                test_total += labels.size(0)

        test_accuracy = test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)

        print(f"{self.name} - Test Results:")
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        return test_accuracy, avg_test_loss
    