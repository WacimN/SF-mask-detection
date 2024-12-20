import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ========================
# Constants and Parameters
# ========================
BATCH_SIZE = 32 # Number of images in each batch
TRAIN_VAL_SPLIT = 0.8 # 80% for training, 20% for validation
DATASET_PROPORTION = 0.1  # Use a fraction of the dataset for faster training
# MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]  # Normalization parameters
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# ========================
# Transforms
# ========================
def get_transforms(augment=False):
    """Return transforms for data preprocessing."""
    if augment:
        print("Data augmentation enabled")
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomAffine(degrees=20, scale=(0.8, 1.2), translate=(0.2, 0.2)),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    
# ========================
# Dataset and Dataloader
# ========================
def load_dataset(data_dir, proportion=DATASET_PROPORTION, augment=False):
    """
    Load a dataset with optional data augmentation and limit its size by proportion.

    Args:
        data_dir (str): Path to the dataset directory.
        proportion (float): Proportion of the dataset to use.
        augment (bool): Whether to apply data augmentation.

    Returns:
        Tuple[torchvision.datasets.ImageFolder, torch.utils.data.Subset]: Full dataset and subset.
    """
    transform = get_transforms(augment=augment)
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Select only a proportion of the dataset
    subset_size = int(len(dataset) * proportion)
    subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

    return dataset, subset


def split_train_val(train_dataset, split_ratio=TRAIN_VAL_SPLIT):
    """
    Split the training dataset into training and validation sets.
    
    Args:
        train_subset: Subset of the training dataset.
        split_ratio (float): Ratio for splitting training and validation datasets.
    
    Returns:
        Tuple[torch.utils.data.Subset, torch.utils.data.Subset]: Training and validation subsets.
    """
    train_size = int(len(train_dataset) * split_ratio)
    val_size = len(train_dataset) - train_size
    return random_split(train_dataset, [train_size, val_size]) #train_subset, val_subset = random_split(train_subset, [train_size, val_size])


def get_dataloaders(train_dir, test_dir, augment=False, batch_size=BATCH_SIZE):
    """
    Create DataLoaders for training, validation, and testing datasets.
    
    Args:
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the testing data directory.
        augment (bool): Whether to apply data augmentation to the training set.
        batcg_size (int): Number of images in each batch.
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
    """
    # Load datasets with 10% of the data
    train_dataset, train_subset = load_dataset(train_dir, proportion=DATASET_PROPORTION, augment=True)
    test_dataset, test_subset = load_dataset(test_dir, augment=False)

    # Split train_subset into train and validation
    train_subset, val_subset = split_train_val(train_subset)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_dataset, train_subset, val_subset , test_subset, train_loader, val_loader, test_loader

