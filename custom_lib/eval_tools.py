import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from poutyne import Model
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


def bootstrap_evaluation_poutyne(model, test_loader, save_logs, n_bootstraps, seed, results_dir=None):
    """
    Perform bootstrap evaluation of a model on a test dataset.

    Args:
        model: The trained Poutyne model to evaluate.
        test_loader: DataLoader for the test dataset.
        save_logs: Whether to save the metric distributions to CSV.
        results_dir: Directory to save the bootstrap distribution CSV.
        n_bootstraps: Number of bootstrap samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        A pandas DataFrame with mean and confidence intervals for:
        - Accuracy
        - F1 Score
        - Sensitivity (Recall)
        - Specificity
        - Test Loss
    """
    rng = np.random.RandomState(seed)

    # Store bootstrapped metrics
    metrics = {
        "accuracy": [],
        "f1_score": [],
        "sensitivity": [],
        "specificity": [],
        "loss": [],
    }

    for _ in range(n_bootstraps):
        sampled_indices = rng.choice(len(test_loader.dataset), len(test_loader.dataset), replace=True)
        sampled_subset = Subset(test_loader.dataset, sampled_indices)
        sampled_loader = DataLoader(sampled_subset, batch_size=test_loader.batch_size, shuffle=False)

        # Evaluate using Poutyne
        test_loss, test_acc = model.evaluate_generator(sampled_loader)

        # Extract predictions and true labels
        y_true, y_pred = [], []
        for inputs, labels in sampled_loader:
            outputs = model.predict_on_batch(inputs)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(np.argmax(outputs, axis=1))

        # Compute metrics
        f1 = f1_score(y_true, y_pred, average='macro')
        sensitivity = recall_score(y_true, y_pred, average='macro')

        # Compute specificity
        cm = confusion_matrix(y_true, y_pred)
        specificity_values = []
        for i in range(cm.shape[0]):
            col_sum = cm[:, i].sum()
            if col_sum > 0:
                specificity_values.append(cm[i, i] / col_sum)
        specificity = np.mean(specificity_values) if specificity_values else 0.0

        # Store results
        metrics["accuracy"].append(test_acc)
        metrics["f1_score"].append(f1)
        metrics["sensitivity"].append(sensitivity)
        metrics["specificity"].append(specificity)
        metrics["loss"].append(test_loss)

    if save_logs:
        # Save the full bootstrap distributions
        dist_df = pd.DataFrame(metrics)
        dist_df.to_csv(f"{results_dir}/bootstrap_distribution.csv", index=False)

    # Compute mean and confidence intervals
    def compute_ci(values):
        return np.mean(values), np.percentile(values, 2.5), np.percentile(values, 97.5)

    results_dict = {f"boot_{metric}_{stat}": value
                    for metric, values in metrics.items()
                    for stat, value in zip(["mean", "low", "high"], compute_ci(values))}

    # Convert to DataFrame
    results_df = pd.DataFrame([results_dict])

    return results_df




def evaluate_tb_class(model, test_loader, tb_class_index):
    """
    Evaluates the sensitivity and specificity for the TB class.
    
    Args:
        model: The trained Poutyne model.
        test_loader: DataLoader for the test dataset.
        tb_class_index: The index of the TB class in the class_names list.
        
    Returns:
        A dictionary with sensitivity and specificity for the TB class.
    """
    # Store true labels and predictions
    y_true, y_pred = [], []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(model.device), labels.to(model.device)  # Move to GPU if available
        outputs = model.predict_on_batch(inputs)  # Get model predictions
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(np.argmax(outputs, axis=1))  # Convert logits to class indices

    # Compute confusion matrix for TB class
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[tb_class_index, tb_class_index]  # True positives for TB class
    fn = cm[tb_class_index].sum() - tp  # False negatives for TB class
    fp = cm[:, tb_class_index].sum() - tp  # False positives for TB class
    tn = cm.sum() - (tp + fn + fp)  # True negatives for TB class

    # Calculate sensitivity (recall) for the TB class
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Calculate specificity for the TB class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Return the results
    results = {sensitivity, specificity}

    return results


def tb_metrics_generator(y_pred, y_true, tb_class_index):
    """
    Generate sensitivity and specificity for the TB class, given the predicted and true labels.

    Args:
        y_pred: The predicted labels (model output, probability values or logits).
        y_true: The true labels.
        tb_class_index: The index of the TB class (default is 1 for TB).

    Returns:
        sensitivity: Sensitivity for the TB class.
        specificity: Specificity for the TB class.
    """
    
    # Convert predictions to class labels (assuming y_pred is the raw output, e.g., logits or probabilities)
    y_pred_to_class = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_to_class)

    # Define indices for the classes
    if tb_class_index == 0:
        # TB is the first class (index 0)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    else:
        # TB is the second class (index 1)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # Calculate TB sensitivity (Recall)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Avoid division by zero

    # Calculate TB specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero

    return sensitivity, specificity