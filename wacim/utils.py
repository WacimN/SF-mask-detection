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
    plt.title('Confusion Matrix')
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
import torch.nn as nn
from torchvision import models

def initialize_model(model_class, num_classes, device="cuda:0"):
    """
    Initialize a model with a specified number of output classes.

    Args:
        model_class: Model class to initialize (e.g., models.mobilenet_v2 or models.efficientnet_b0).
        num_classes: Number of output classes.
        device: Device to load the model (default is "cuda:0").

    Returns:
        Initialized model.
    """
    model = model_class(pretrained=True)

    # Check model class and adjust the final layer
    if isinstance(model, models.MobileNetV2):
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif isinstance(model, models.EfficientNet):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model class")

    return model.to(device)



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
