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
