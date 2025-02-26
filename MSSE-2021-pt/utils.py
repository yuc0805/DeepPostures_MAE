import csv
import torch
import numpy as np


# Function to write metrics to CSV
def write_metrics_to_csv(metrics, csv_file):
    """
    Write training and validation metrics to a CSV file.

    Parameters:
    - metrics (list of dict): List of dictionaries containing epoch, train_loss, train_acc, val_loss, and val_acc.
    - csv_file (str): The name of the CSV file to write to.
    """
    if not metrics:
        return
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    print(f"Metrics written to {csv_file}")


# Function to load model weights
def load_model_weights(model, file_path, weights_only=True):
    model.load_state_dict(torch.load(file_path, weights_only=weights_only))
    print(f"Weights loaded from {file_path}")


def compute_accuracy_from_confusion_matrix(cm):
    # True Positives (TP): Diagonal elements
    true_positive = np.diag(cm)

    # False Positives (FP): Column sum - TP
    false_positive = cm.sum(axis=0) - true_positive

    # False Negatives (FN): Row sum - TP
    false_negative = cm.sum(axis=1) - true_positive

    # True Negatives (TN): Total sum - (TP + FP + FN)
    total = cm.sum()
    true_negative = total - (true_positive + false_positive + false_negative)

    # Accuracy: Total correct predictions / Total samples
    accuracy = true_positive.sum() / total

    # Sensitivity (Recall): TP / (TP + FN)
    sensitivity = true_positive / (true_positive + false_negative)
    
    # Specificity: TN / (TN + FP)
    specificity = true_negative / (true_negative + false_positive)

    # Balanced Accuracy: Average of Sensitivity and Specificity
    balanced_accuracy = np.mean((sensitivity + specificity) / 2)

    return accuracy, balanced_accuracy

def compute_additional_metrics_from_confusion_matrix(cm):
    """
    Compute specificity, sensitivity, positive predictive value (PPV),
    and negative predictive value (NPV) from a multi-class confusion matrix.

    :param cm: NumPy array representing the confusion matrix (square matrix).
    :return: Dictionary containing computed metrics for each class.
    """
    # True Positives (TP): Diagonal elements
    true_positive = np.diag(cm)

    # False Positives (FP): Column sum - TP
    false_positive = cm.sum(axis=0) - true_positive

    # False Negatives (FN): Row sum - TP
    false_negative = cm.sum(axis=1) - true_positive

    # True Negatives (TN): Total sum - (TP + FP + FN)
    total = cm.sum()
    true_negative = total - (true_positive + false_positive + false_negative)

    # Compute metrics
    sensitivity = true_positive / (true_positive + false_negative)  # Recall
    specificity = true_negative / (true_negative + false_positive)
    ppv = true_positive / (true_positive + false_positive)  # Precision
    npv = true_negative / (true_negative + false_negative)

    # Handle division by zero
    sensitivity = np.nan_to_num(sensitivity)
    specificity = np.nan_to_num(specificity)
    ppv = np.nan_to_num(ppv)
    npv = np.nan_to_num(npv)

    # Return metrics for each class
    return {
        "sensitivity": sensitivity.tolist(),
        "specificity": specificity.tolist(),
        "positive_predictive_value": ppv.tolist(),
        "negative_predictive_value": npv.tolist()
    }
