
import json
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import datetime
from itertools import product
from torch import nn
from PIL import Image
from tqdm import tqdm
import time
from torchvision import models
import numpy as np
import random
import pickle

from torchvision import transforms
from data_load import MEAN, STD

from sklearn.metrics import roc_curve, auc




# ========================
# Data Preprocessing
# ======================== 

def pad_and_resize(input_dir, output_dir, target_size=(224, 224), padding_color=(0, 0, 0)):
    """
    Resize images with aspect ratio preservation and pad to target size.

    Args:
        input_dir (str): Path to input directory containing image categories.
        output_dir (str): Path to save resized and padded images.
        target_size (tuple): Target size (width, height).
        padding_color (tuple): Padding color as (R, G, B).

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)
        
        for filename in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
            input_path = os.path.join(category_path, filename)
            output_path = os.path.join(output_category_path, filename)
            try:
                with Image.open(input_path) as img:
                    old_size = img.size  # (width, height)
                    ratio = float(target_size[0]) / max(old_size)
                    new_size = tuple([int(x * ratio) for x in old_size])
                    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    new_img = Image.new("RGB", target_size, padding_color)  # Use specified padding color
                    new_img.paste(
                        img_resized,
                        ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2),
                    )
                    new_img.save(output_path)
            except Exception as e:
                print(f"Error with {input_path}: {e}")
                

# ========================
# Metrics Saving
# ========================

def save_metrics_to_json(metrics, filename, k_folds=None, hyperparam_search=None):
    """
    Save training metrics to a JSON file with additional information.
    
    Args:
        metrics: Dictionary containing training metrics.
        filename: Name of the JSON file.
        k_folds: (Optional) Number of folds used in cross-validation.
        hyperparam_search: (Optional) Boolean indicating if hyperparameter search is used.
    """
    metrics_data = {
        "metrics": metrics,
        "k_folds": k_folds,
        "hyperparameter_search": hyperparam_search,
    }

    try:
        # Load existing data if the file exists
        with open(filename, "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []

    # Append new metrics and save
    existing_data.append(metrics_data)
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)
    print(f"Metrics saved to {filename}")



def save_plot(fig, plot_name):
    save_path = f"results/plots/{plot_name}.html"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)


# ========================
# Metrics Visualization
# ========================

def plot_loss(epochs, train_losses, val_losses):    
    # Plot Training and Validation Loss
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses,
        mode='lines+markers',
        name='Training Loss'
    ))
    loss_fig.add_trace(go.Scatter(
        x=epochs,
        y=val_losses,
        mode='lines+markers',
        name='Validation Loss'
    ))
    loss_fig.update_layout(title="Loss Across Epochs",
                        xaxis_title="Epoch",
                        yaxis_title="Loss")
    # Save the loss figure as an HTML file
    loss_fig.write_html("./training/training_metrics_plot/loss_plot.html")
    loss_fig.show()

# Plot Training and Validation Accuracy
def plot_accuracy(epochs, train_accuracies, val_accuracies, training_time_name):
    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(go.Scatter(
        x=epochs,
        y=train_accuracies,
        mode='lines+markers',
        name='Training Accuracy'
    ))
    accuracy_fig.add_trace(go.Scatter(
        x=epochs,
        y=val_accuracies,
        mode='lines+markers',
        name='Validation Accuracy'
    ))
    accuracy_fig.update_layout(title="Accuracy Across Epochs",
                            xaxis_title="Epoch",
                            yaxis_title="Accuracy")
    # Save the accuracy figure as an HTML file
    accuracy_fig.write_html(f"./training/training_metrics_plot/accuracy_plot_{training_time_name}.html")
    accuracy_fig.show()

def plot_time(epochs, times, training_time_name):
    # Plot Training Time per Epoch
    time_fig = go.Figure()
    time_fig.add_trace(go.Scatter(
        x=epochs,
        y=times,
        name='Training Time'
    ))
    time_fig.update_layout(title="Training Time per Epoch",
                        xaxis_title="Epoch",
                        yaxis_title="Time (seconds)")
    # Save the time figure as an HTML file
    time_fig.write_html(f"training/training_metrics_plot/time_plot_{training_time_name}.html")
    time_fig.show()
    
    
def plot_confusion_matrix(model, test_loader, classes, device):
    y_true, y_pred = [], []
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix {model.__class__.__name__}, test dataset size: {len(y_true)}')
    plt.show()
    print(classification_report(y_true, y_pred, target_names=classes))
    


def load_results(results_file):
    """
    Load existing results from a JSON file or return an empty list if the file doesn't exist.
    
    Args:
        results_file (str): Path to the results JSON file.
    
    Returns:
        list: List of results.
    """
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []  # Return an empty list if file doesn't exist or is corrupted


def save_results(results, results_file):
    """
    Save results to a JSON file.
    
    Args:
        results (list): List of results to save.
        results_file (str): Path to the results JSON file.
    """
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")


def generate_param_grid(learning_rates, batch_sizes):
    """
    Generate a grid of hyperparameter combinations.
    
    Args:
        learning_rates (list): List of learning rates.
        batch_sizes (list): List of batch sizes.
    
    Returns:
        list: List of hyperparameter combinations as tuples.
    """
    return list(product(learning_rates, batch_sizes))


def get_current_timestamp():
    """
    Get the current timestamp in a formatted string.
    
    Returns:
        str: Current timestamp in "YYYY-MM-DD_HH-MM-SS" format.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_best_config(results):
    """
    Find and display the best configuration based on validation accuracy.
    
    Args:
        results (list): List of results.
    
    Returns:
        dict: Best configuration.
    """
    best_config = max(results, key=lambda x: x["val_accuracy"])
    print(f"\nBest configuration: lr={best_config['lr']}, batch_size={best_config['batch_size']} "
          f"with val_accuracy={best_config['val_accuracy']:.2f}%")
    return best_config

# ========================
# Helper Functions
# ========================

def load_model(model_name, num_classes, model_save_path, device="cuda:0"):
    """
    Load a pre-trained model and its weights from a .pth file.

    Args:
        model_name (str): The name of the model to load (e.g., "mobilenet_v2", "efficientnet_b0").
        num_classes (int): Number of output classes for the model.
        model_save_path (str): Path to the .pth file containing the trained weights.
        device (str): Device to load the model on (default: "cuda:0").

    Returns:
        model: The loaded and prepared model.
    """
    # Initialize the model
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name.startswith("efficientnet"):
        model = getattr(models, model_name)(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Load the weights
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=device), strict=False)
        print(f"Weights loaded successfully from {model_save_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading weights from {model_save_path}: {e}")

    # Move model to device
    model = model.to(device)

    # Set to evaluation mode
    model.eval()
    print(f"Model {model_name} loaded and set to evaluation mode on device: {device}")

    return model


def save_best_model(current_accuracy, best_accuracy, model, save_path):
    """
    Save the model if it has the best accuracy so far.

    Args:
        current_accuracy: Current validation accuracy.
        best_accuracy: Best validation accuracy so far.
        model: Model to save.
        save_path: Path to save the model.

    Returns:
        Updated best accuracy.
    """
    if current_accuracy > best_accuracy:
        torch.save(model.state_dict(), save_path)
        print(f"New best model saved with accuracy: {current_accuracy:.2f}%")
        return current_accuracy
    return best_accuracy


def measure_inference_time(model, dataloader, device="cuda:0"):
    """
    Measure inference time for a single batch of images.
    
    Args:
        model: The trained model.
        dataloader: A DataLoader for the test data.
        device: Device to perform inference on.
        
    Returns:
        Average inference time per image (ms).
    """
    model.eval()
    total_time = 0
    num_samples = 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)
            num_samples += inputs.size(0)
            break  # Measure on a single batch
    avg_inference_time = (total_time / num_samples) * 1000  # Convert to ms
    return avg_inference_time




def calculate_mean_std(metrics_list, test_accuracy):
    """
    Calculate mean and standard deviation for train/validation metrics and include test accuracy.

    Args:
        metrics_list (list of dicts): List of training metrics for each epoch.
        test_accuracy (float): Test accuracy for the model.

    Returns:
        dict: Dictionary containing mean and std for each metric.
    """
    results = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": []
    }

    for metrics in metrics_list:
        results["train_loss"].append(metrics["train_loss"])
        results["val_loss"].append(metrics["val_loss"])
        results["train_accuracy"].append(metrics["train_accuracy"])
        results["val_accuracy"].append(metrics["val_accuracy"])
        results["train_f1"].append(metrics["train_f1"])
        results["val_f1"].append(metrics["val_f1"])

    summary = {
        "train_loss": (np.mean(results["train_loss"]), np.std(results["train_loss"])),
        "val_loss": (np.mean(results["val_loss"]), np.std(results["val_loss"])),
        "train_accuracy": (np.mean(results["train_accuracy"]), np.std(results["train_accuracy"])),
        "val_accuracy": (np.mean(results["val_accuracy"]), np.std(results["val_accuracy"])),
        "train_f1": (np.mean(results["train_f1"]), np.std(results["train_f1"])),
        "val_f1": (np.mean(results["val_f1"]), np.std(results["val_f1"])),
        "test_accuracy": test_accuracy
    }

    return summary


def plot_roc_curve(model, test_loader, num_classes, model_name, device="cuda:0"):
    """
    Plot the ROC curve for a model using the test data.
    
    Args:
        model: The trained model.
        test_loader: DataLoader for the test dataset.
        num_classes: Number of output classes.
        model_name: Name of the model for labeling the plot.
        device: Device for computation (default: "cuda:0").
    """
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            y_scores.extend(probabilities[:, 1].cpu().numpy())  # Score for the positive class
            y_true.extend(labels.cpu().numpy())

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    return roc_auc



def get_random_image_from_test_dir(test_dir):
    """Get a random image path from the test directory."""
    image_paths = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                image_paths.append(os.path.join(root, file))
    return random.choice(image_paths)

def visualize_augmentations(image_path, transform, n_augmentations=5):
    """Visualize augmentations on the same image."""
    original_image = Image.open(image_path).convert("RGB")

    # Define augmentation names
    augmentation_names = [
        "Horizontal Flip",
        "Rotation",
        "Color Jitter",
        "Perspective Distortion",
        "Affine Transformation"
    ]

    # Apply augmentations
    augmented_images = [transform(original_image) for _ in range(n_augmentations)]

    # Convert tensors back to PIL images for visualization
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[-m / s for m, s in zip(MEAN, STD)], std=[1 / s for s in STD]),
        transforms.ToPILImage()
    ])
    
    augmented_images = [inv_transform(img) for img in augmented_images]

    # Plot original and augmented images
    fig, axes = plt.subplots(1, n_augmentations + 1, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for i, aug_img in enumerate(augmented_images):
        axes[i + 1].imshow(aug_img)
        axes[i + 1].set_title(augmentation_names[i % len(augmentation_names)])
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()
    
def visualize_individual_augmentations(image_path):
    """Visualize the effect of each augmentation separately on the original image."""
    original_image = Image.open(image_path).convert("RGB")

    # Define individual transformations with names
    transformations = [
        ("Horizontal Flip", transforms.RandomHorizontalFlip(p=1.0)),
        ("Rotation", transforms.RandomRotation(degrees=15)),
        ("Color Jitter", transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2)),
        ("Perspective Distortion", transforms.RandomPerspective(distortion_scale=0.2, p=1.0)),
        ("Affine Transformation", transforms.RandomAffine(degrees=20, scale=(0.8, 1.2), translate=(0.2, 0.2))),
    ]

    # Apply each transformation individually
    augmented_images = [(name, trans(original_image)) for name, trans in transformations]

    # Plot original and augmented images
    fig, axes = plt.subplots(1, len(augmented_images) + 1, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for i, (name, img) in enumerate(augmented_images):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(name)
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()
    
    

def save_metrics_and_models(model_metrics, model_training_time,model_path, model_name):
    
    # Créer le dossier 'saved_notebook_models' s'il n'existe pas déjà
    os.makedirs('saved_notebook_models', exist_ok=True)

    # Enregistrer les métriques et les temps d'entraînement
    with open(f'saved_notebook_models/{model_name}_metrics.pkl', 'wb') as f:
        pickle.dump(model_metrics, f)

    with open(f'saved_notebook_models/{model_name}_training_time.pkl', 'wb') as f:
        pickle.dump(model_training_time, f)

    # Enregistrer les chemins des modèles
    with open(f'saved_notebook_models/{model_name}_model_path.txt', 'w') as f:
        f.write(model_path)

def load_metrics_and_models(model_name):
    # Charger les métriques et les temps d'entraînement
    with open(f'saved_notebook_models/{model_name}ç_metrics.pkl', 'rb') as f:
        model_metrics = pickle.load(f)

    with open(f'saved_notebook_models/{model_name}_training_time.pkl', 'rb') as f:
        model_training_time = pickle.load(f)

    # Charger les chemins des modèles
    with open(f'saved_notebook_models/{model_name}_model_path.txt', 'r') as f:
        model_path = f.read()

    return model_metrics, model_training_time, model_path


# Updated plot_accuracy_models function
def plot_accuracy_models(metrics1, metrics2, model1_name, model2_name, lr, batch_size):
    epochs1 = list(range(1, len(metrics1) + 1))
    epochs2 = list(range(1, len(metrics2) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs1, y=[m["train_accuracy"] for m in metrics1], mode='lines+markers', name=f"{model1_name} Train Accuracy"))
    fig.add_trace(go.Scatter(x=epochs1, y=[m["val_accuracy"] for m in metrics1], mode='lines+markers', name=f"{model1_name} Val Accuracy"))
    fig.add_trace(go.Scatter(x=epochs2, y=[m["train_accuracy"] for m in metrics2], mode='lines+markers', name=f"{model2_name} Train Accuracy"))
    fig.add_trace(go.Scatter(x=epochs2, y=[m["val_accuracy"] for m in metrics2], mode='lines+markers', name=f"{model2_name} Val Accuracy"))
    fig.update_layout(
        title=f"Training vs Validation Accuracy (lr={lr}, bs={batch_size})",
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)"
    )
    fig.show()

# Updated plot_loss_models function
def plot_loss_models(metrics1, metrics2, model1_name, model2_name, lr, batch_size):
    epochs1 = list(range(1, len(metrics1) + 1))
    epochs2 = list(range(1, len(metrics2) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs1, y=[m["train_loss"] for m in metrics1], mode='lines+markers', name=f"{model1_name} Train Loss"))
    fig.add_trace(go.Scatter(x=epochs1, y=[m["val_loss"] for m in metrics1], mode='lines+markers', name=f"{model1_name} Val Loss"))
    fig.add_trace(go.Scatter(x=epochs2, y=[m["train_loss"] for m in metrics2], mode='lines+markers', name=f"{model2_name} Train Loss"))
    fig.add_trace(go.Scatter(x=epochs2, y=[m["val_loss"] for m in metrics2], mode='lines+markers', name=f"{model2_name} Val Loss"))
    fig.update_layout(
        title=f"Training vs Validation Loss (lr={lr}, bs={batch_size})",
        xaxis_title="Epoch",
        yaxis_title="Loss"
    )
    fig.show()
