import time
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, ParameterGrid
from torch import nn
from utils import save_best_model, save_metrics_to_json
import os
from torchvision import models


# ========================
# Initialize Model Function
# ========================

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

# ========================
# Train Model Function (Updated)
# ========================
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs, training_time_name, device="cuda:0", use_early_stopping=True):
    print(f"size of train_loader : {len(train_loader)}, size of val_loader : {len(val_loader)}")
    print(f"Training {model.__class__.__name__} for {epochs} epochs...")
    training_metrics = []
    start_time = time.time()

    # Initialize early stopping if enabled
    early_stopping = EarlyStopping(patience=5, delta=0) if use_early_stopping else None

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        # Training loop
        for inputs, labels in train_loader:
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")  # Debugging dimensions

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate training accuracy and collect predictions/labels
            _, predicted = torch.max(outputs, 1)
            # predicted = (outputs >= 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        train_accuracy = 100 * correct / total
        train_f1 = f1_score(all_labels, all_predictions, average="weighted")
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_loader)

        # Validation metrics
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_predictions = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # labels = labels.unsqueeze(1)  # Ajuste la forme new
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_labels.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_f1 = f1_score(val_labels, val_predictions, average="weighted")

        # Store epoch metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "time": epoch_time,
            "learning_rate": optimizer.param_groups[0]["lr"]  # Learning rate
        }
        
        training_metrics.append(epoch_metrics)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Train F1: {train_f1:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, "
              f"Val F1: {val_f1:.4f}, Time: {epoch_time:.2f}s")

        # Early stopping check
        if use_early_stopping:
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f}s")

    # Save model weights after training
    model_save_path = f"training/{model.__class__.__name__}_{training_time_name}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")

    return training_metrics, model_save_path, total_time




# ========================
# K-Fold Cross-Validation with Hyperparameter Tuning
# ========================
def kfold_with_hyperparam_tuning(
    model_class, original_dataset, dataset, criterion, optimizer_class, param_grid, augment_strategies, k_folds=5, epochs=30, device="cuda:0", results_file = "kfold_hyperparam_results.json"
):
    """
    Perform K-fold cross-validation with hyperparameter tuning and data augmentation comparison.

    Args:
        model_class: Class of the model to initialize.
        dataset: Full dataset for K-fold splitting.
        criterion: Loss function.
        optimizer_class: Optimizer class (e.g., optim.Adam).
        param_grid: Grid of hyperparameters.
        augment_strategies: List of booleans specifying augmentation settings.
        k_folds: Number of folds for K-fold CV.
        epochs: Number of epochs per training.
        device: Device for training (default "cuda:0").

    Returns:
        Aggregated metrics for all configurations.
    """
    print(f"Running K-fold cross-validation with hyperparameter tuning for {model_class.__name__}...\n")
    print(f"{len(dataset)} samples in the dataset = {len(original_dataset)}/{len(dataset)}= {len(original_dataset)/len(dataset)}%. {k_folds}-fold CV with {epochs} epochs per fold.")
    param_combinations = list(ParameterGrid(param_grid))
    all_results = []
    
    classes =  original_dataset.classes

    os.makedirs("results", exist_ok=True)

    for augment in augment_strategies:
        augment_desc = "With Augmentation" if augment else "Without Augmentation"
        print(f"\n=== Running {augment_desc} ===")

        for params in param_combinations:
            print(f"\nTesting hyperparameters: {params}")

            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            best_accuracy = 0.0
            fold_results = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
                print(f"\nFold {fold + 1}/{k_folds} for {params}")

                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)

                train_loader = DataLoader(
                    train_subset,
                    batch_size=params["batch_size"],
                    shuffle=True,
                    collate_fn=None if augment else lambda x: x,  # Apply augmentation conditionally
                )
                val_loader = DataLoader(val_subset, batch_size=params["batch_size"], shuffle=False)

                # Initialize model and optimizer
                model = initialize_model(model_class, num_classes=len(classes), device=device)
                optimizer = optimizer_class(model.parameters(), lr=params["learning_rate"])

                # Train model
                training_time_name = f"lr_{params['learning_rate']}_bs_{params['batch_size']}_aug_{augment}_fold_{fold+1}"
                training_metrics, model_save_path, total_time = train_model(
                    model, train_loader, val_loader, None, criterion, optimizer, epochs, training_time_name, device=device
                )

                # Track best model for this configuration
                val_accuracy = training_metrics[-1]["val_accuracy"]
                best_accuracy = save_best_model(val_accuracy, best_accuracy, model, f"results/best_model_{round(val_accuracy,2)}_{training_time_name}.pth")

                fold_results.append({
                "fold": fold + 1,
                "model_save_path": model_save_path,
                "total_time_training": total_time,
                "val_accuracy": val_accuracy,
                "training_metrics": training_metrics,
            })
            
            all_total_time = sum([result["total_time_training"] for result in fold_results])

            all_results.append({
                "all_total_time_kfold": all_total_time,
                "params": params,
                "augment": augment,
                "fold_results": fold_results,
            })

    return all_results

# def kfold_cross_validation(model, dataset, criterion, optimizer, epochs, k_folds=5, training_time_name="model"):
#     # Création du KFold
#     kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
#     fold_metrics = []  # Pour stocker les métriques de chaque pli

#     # Diviser le dataset en k plis
#     for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
#         print(f"Training for fold {fold+1}/{k_folds}")
        
#         # Diviser le dataset en train et validation pour chaque pli
#         train_subset = Subset(dataset, train_idx)
#         val_subset = Subset(dataset, val_idx)

#         # Créer des DataLoaders pour chaque pli
#         train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
#         val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

#         # Appeler la fonction de formation pour chaque pli
#         training_metrics, model_save_path, total_time = train_model(
#             model=model, 
#             train_loader=train_loader, 
#             val_loader=val_loader, 
#             test_loader=None,  # Il n'est pas utilisé ici
#             criterion=criterion, 
#             optimizer=optimizer, 
#             epochs=epochs, 
#             training_time_name=f"{training_time_name}_fold_{fold+1}"
#         )

#         # Sauvegarder les métriques du pli
#         fold_metrics.append({
#             "fold": fold + 1,
#             "training_metrics": training_metrics,
#             "model_save_path": model_save_path
#         })

#     return fold_metrics



# Evaluate Model
def evaluate_model(model, test_loader, device = "cuda:0"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


# ========================
# Early Stopping
# ========================
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
                
def initialize_model_sigmoid(model_class, device="cuda:0"):
    """
    Initialize a model for binary classification.

    Args:
        model_class: Model class to initialize (e.g., models.mobilenet_v2 or models.efficientnet_b0).
        device: Device to load the model (default is "cuda:0").

    Returns:
        Initialized model.
    """
    model = model_class(pretrained=True)

    # Check model class and adjust the final layer
    if isinstance(model, models.MobileNetV2):
        model.classifier[1] = nn.Sequential(
            nn.Linear(model.last_channel, 1),
            nn.Sigmoid()
        )
    elif isinstance(model, models.EfficientNet):
        model.classifier[1] = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 1),
            nn.Sigmoid()
        )
    else:
        raise ValueError("Unsupported model class")

    return model.to(device)