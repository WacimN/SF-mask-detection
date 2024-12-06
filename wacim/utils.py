import json
import plotly.graph_objects as go


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