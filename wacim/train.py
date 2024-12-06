import time
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold



# ========================
# Train Model Function (Updated)
# ========================
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs, training_time_name, device = "cuda:0"):
    print(f"size of train_loader : {len(train_loader)}, size of val_loader : {len(val_loader)}")
    print(f"Training {model.__class__.__name__} for {epochs} epochs...")
    training_metrics = []
    start_time = time.time()

    # Initialisation de l'early stopping
    early_stopping = EarlyStopping(patience=5, delta=0)

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
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate training accuracy and collect predictions/labels
            _, predicted = torch.max(outputs, 1)
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
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f}s")

    # Save model weights after training
    model_save_path = f"training/mobilenetv2_trained_{training_time_name}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")

    return training_metrics, model_save_path, total_time



def kfold_cross_validation(model, dataset, criterion, optimizer, epochs, k_folds=5, training_time_name="model"):
    # Création du KFold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_metrics = []  # Pour stocker les métriques de chaque pli

    # Diviser le dataset en k plis
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Training for fold {fold+1}/{k_folds}")
        
        # Diviser le dataset en train et validation pour chaque pli
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Créer des DataLoaders pour chaque pli
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # Appeler la fonction de formation pour chaque pli
        training_metrics, model_save_path, total_time = train_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            test_loader=None,  # Il n'est pas utilisé ici
            criterion=criterion, 
            optimizer=optimizer, 
            epochs=epochs, 
            training_time_name=f"{training_time_name}_fold_{fold+1}"
        )

        # Sauvegarder les métriques du pli
        fold_metrics.append({
            "fold": fold + 1,
            "training_metrics": training_metrics,
            "model_save_path": model_save_path
        })

    return fold_metrics



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