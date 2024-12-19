# Inspiration https://github.com/Piyush2912/Real-Time-Face-Mask-Detection/blob/main/Code/train_mask_detector.py
# run python train_from_scratch.py -d ../SF-MASK-dataset-padded/train -p plot_from_scratch.png -m mask_detector_from_scratch.pth

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from imutils import paths
import random
from imutils import paths



# Custom dataset for loading images
class MaskDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            # Charger l'image
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Could not load image {image_path}: {e}")
            raise ValueError(f"Invalid image file at {image_path}")

        if self.transform:
            image = self.transform(image)

        return image, label


# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="mask_detector.pth", help="path to output face mask detector model")
args = vars(ap.parse_args())

# Hyperparameters
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Charger les chemins d'images
print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))

if len(image_paths) == 0:
    raise ValueError(f"No images found in the dataset directory: {args['dataset']}")

# Mélanger les chemins d'images et prendre 10%
random.shuffle(image_paths)
num_images = int(len(image_paths) * 0.1)  # 10% des images
image_paths = image_paths[:num_images]

print(f"[INFO] Selected {len(image_paths)} images (10% of the dataset).")

# Charger les images et extraire les labels
data = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    try:
        image = Image.open(image_path).convert("RGB")
        data.append(image)
        labels.append(label)
    except Exception as e:
        print(f"[ERROR] Failed to process image {image_path}: {e}")
        continue

# Vérifier les labels
print(f"[INFO] Unique labels found: {set(labels)}")
if len(labels) == 0:
    raise ValueError("No labels were found. Check dataset directory structure.")

    
# Encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = np.argmax(labels, axis=1)  # Convert one-hot to integers

# Train/test split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Image transformations (with data augmentation)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders
train_dataset = MaskDataset(trainX, trainY, transform=transform_train)
test_dataset = MaskDataset(testX, testY, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

# Define MobileNetV2 architecture from scratch
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            # Add blocks for MobileNetV2 here manually or by using a predefined MobileNetV2 implementation
            # For simplicity, we will simulate with a few Conv layers
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize model, loss function, and optimizer
print("[INFO] initializing MobileNetV2 model from scratch...")
model = MobileNetV2(num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=INIT_LR)

# Train the model
print("[INFO] training model...")
train_loss = []
val_loss = []
train_acc = []
val_acc = []

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    total_train_correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_correct += (outputs.argmax(1) == labels).sum().item()

    train_loss.append(total_train_loss / len(train_loader))
    train_acc.append(total_train_correct / len(train_dataset))

    model.eval()
    total_val_loss = 0
    total_val_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            total_val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss.append(total_val_loss / len(test_loader))
    val_acc.append(total_val_correct / len(test_dataset))

    print(f"Epoch {epoch + 1}/{EPOCHS}: Train Loss={train_loss[-1]:.4f}, Train Acc={train_acc[-1]:.4f}, Val Loss={val_loss[-1]:.4f}, Val Acc={val_acc[-1]:.4f}")

# Save the model
print("[INFO] saving model...")
torch.save(model.state_dict(), args["model"])

# Evaluate the model
print("[INFO] evaluating model...")
model.eval()
all_preds = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        true_labels.extend(labels.numpy())

print(classification_report(true_labels, all_preds, target_names=lb.classes_))

# Plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(range(EPOCHS), train_loss, label="Train Loss")
plt.plot(range(EPOCHS), val_loss, label="Val Loss")
plt.plot(range(EPOCHS), train_acc, label="Train Acc")
plt.plot(range(EPOCHS), val_acc, label="Val Acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
