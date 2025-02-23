import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models as models
import time
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from thop import profile
import argparse
from poutyne import Model, CSVLogger, ModelCheckpoint, EarlyStopping, plot_history, ReduceLROnPlateau, Callback
from custom_lib.data_prep import data_transformation_pipeline, data_loader



# Define the model mapping as a constant (outside the function)
MODEL_MAPPING = {
    "b0": ("efficientnet_b0", models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "b1": ("efficientnet_b1", models.EfficientNet_B1_Weights.IMAGENET1K_V1),
    "b2": ("efficientnet_b2", models.EfficientNet_B2_Weights.IMAGENET1K_V1),
    "b3": ("efficientnet_b3", models.EfficientNet_B3_Weights.IMAGENET1K_V1),
}

def load_efficientnet(model_name, model_mapping, pretrained, seed):
    """
    Load an EfficientNet model based on the provided model name and model mapping.
    """
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models are: {list(model_mapping.keys())}")

    model_class_name, weights = model_mapping[model_name]
    model_class = getattr(models, model_class_name)

    if pretrained:
        effnet = model_class(weights=weights)
    else:
        torch.manual_seed(seed)
        effnet = model_class(weights=None)
    return effnet

class TruncatedEffNet(nn.Module):
    def __init__(self, effnet, num_classes, removed_layers, batch_size, image_size):
        super(TruncatedEffNet, self).__init__()

        # Truncate the EfficientNet backbone
        layers = 7 - removed_layers
        self.effnet_truncated = nn.Sequential(*list(effnet.features.children())[:layers])

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Dynamically calculate the input size for the fully connected layer
        with torch.no_grad():  # Disable gradient tracking for this forward pass
            dummy_input = torch.randn(batch_size, 3, image_size, image_size)  # Example input (batch_size=1, channels=3, height=224, width=224)
            dummy_output = self.effnet_truncated(dummy_input)
            dummy_output = self.global_avg_pool(dummy_output)
            fc_input_size = dummy_output.view(dummy_output.size(0), -1).size(1)  # Flatten and get the size
        
        self.dropout = nn.Dropout(.2)

        # Define the fully connected layer
        self.fc = nn.Linear(fc_input_size, num_classes)

        self.classifier = nn.Sequential(
            nn.Dropout(.2),
            nn.Linear(fc_input_size, num_classes)
                )

    def forward(self, x):
        x = self.effnet_truncated(x)  # Extract features
        x = self.global_avg_pool(x)  # Pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)  # Classification
        return x


def bootstrap_evaluation_poutyne(model, test_loader, save_logs, results_dir, n_bootstraps=1000, seed=42):
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

    results_dict = {f"{metric}_{stat}": value
                    for metric, values in metrics.items()
                    for stat, value in zip(["mean", "low", "high"], compute_ci(values))}

    # Convert to DataFrame
    results_df = pd.DataFrame([results_dict])

    return results_df

class PrintLRSchedulerCallback(Callback):
    def set_model(self, model):
        self.model = model  # Store the model reference

    def on_epoch_end(self, epoch, logs):
        lr = self.model.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}: Current LR = {lr}")

def main(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")


    torch.manual_seed(args.seed)


    try:
        effnet = load_efficientnet(args.model_name, MODEL_MAPPING, args.pretrained, args.seed)
        print(f"Successfully loaded EfficientNet {args.model_name}.")
    except ValueError as e:
        print(e)
        return

    image_size = {"b0": 224, "b1": 240, "b2": 260, "b3": 300}[args.model_name]

    train_transform = data_transformation_pipeline(image_size=image_size, 
                                                   rotate_angle=args.rotate_angle,
                                                   horizontal_flip_prob=args.horizontal_flip_prob,
                                                   gaussian_blur=args.gaussian_blur,
                                                   normalize=args.normalize,
                                                   is_train=True)
    test_transform = data_transformation_pipeline(image_size=image_size, 
                                                  rotate_angle=args.rotate_angle,
                                                  horizontal_flip_prob=args.horizontal_flip_prob,
                                                  gaussian_blur=args.gaussian_blur, 
                                                  normalize=args.normalize,
                                                  is_train=False)
    val_transform = data_transformation_pipeline(image_size=image_size, 
                                                 rotate_angle=args.rotate_angle,
                                                 horizontal_flip_prob=args.horizontal_flip_prob,
                                                 gaussian_blur=args.gaussian_blur, 
                                                 normalize=args.normalize,
                                                 is_train=False)

    train_loader, val_loader, test_loader, num_classes = data_loader(args.data_dir, 
                                                                     train_transform=train_transform,
                                                                     test_transform=test_transform,
                                                                     val_transform=val_transform, 
                                                                     seed=args.seed,
                                                                     batch_size=args.batch_size,
                                                                     train_prop=.9,
                                                                     val_prop=.05)

    model = TruncatedEffNet(effnet, num_classes, removed_layers=args.truncated_layers, batch_size=args.batch_size,
                            image_size=image_size)

    if args.save_logs:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        results_dir = os.path.join(f"{args.results_folder_name}/{args.model_name}_reduced_layers_{args.truncated_layers}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        print(f"Logs and output will be saved in: {results_dir}")
    else:
        results_dir = None

    # Wrap the model with Poutyne
    poutyne_model = Model(
        model,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        loss_function=nn.CrossEntropyLoss(),
        batch_metrics=['accuracy'],
        device=device
    )

    callbacks = [ReduceLROnPlateau(
                        monitor='val_loss',  # Monitor validation loss
                        factor=0.1,          # Reduce LR by a factor of 0.1
                        patience=5          # Wait 5 epochs before reducing LR
                                            ),
                PrintLRSchedulerCallback()]
    
    if args.save_logs:
        callbacks.extend([
            ModelCheckpoint(f"{results_dir}/best_model.pth", monitor='val_loss', mode='min', save_best_only=True),
            CSVLogger(f"{results_dir}/training_logs.csv")
   
            # EarlyStopping(monitor = 'val_loss', patience = 5)
            
                        ])

        

    start_time = time.time()
    history = poutyne_model.fit_generator(train_loader, val_loader, epochs=args.epochs, callbacks=callbacks)
    end_time = time.time()
    run_time = end_time - start_time

    print(f"Model training took {run_time / 60} minutes")

    if args.save_logs:
        best_model_path = f"{results_dir}/best_model.pth"
        
        # Load the state dict into the model
        poutyne_model.network.load_state_dict(torch.load(best_model_path))

        # Plot training history
        plot_history(
            history,
            metrics=['loss', 'acc'],
            labels=['Loss', 'Accuracy'],
            titles=f"{args.model_name} Training",
            save=True,
            save_filename_template='{metric}_plot',
            save_directory=results_dir,
            save_extensions=('png',)
        )

    # Bootstrap evaluation
    print("Starting Bootstrapping")
    boot_strap_results = bootstrap_evaluation_poutyne(model=poutyne_model, test_loader=test_loader, save_logs= args.save_logs, n_bootstraps=args.bootstrap_n, 
                                                      results_dir=results_dir)
    print(boot_strap_results)

    # Compute FLOPs and parameters
    dummy_input = torch.randn(args.batch_size, 3, image_size, image_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    gflops = flops / 1000000000
    print(f"GFLOPs: {gflops}")
    print(f"Parameters: {params}")

    if args.save_logs:
        if os.path.exists(f"{args.results_folder_name}/test_results.csv"):
            test_results_df = pd.read_csv(f"{args.results_folder_name}/test_results.csv")
        else:
            test_results_df = pd.DataFrame(columns=[
                "model_id", "model", "pretrained", "truncated_layers", "epochs", "run_time", "lr", "image_size",
                "rotate_angle", "horizontal_flip_prob", "gaussian_blur", "normalize", "seed", "gflops", "params"
            ])

        new_results_df = pd.DataFrame({
            "model_id": [f"{args.model_name}_reduced_layers_{args.truncated_layers}_{timestamp}"],
            "model": [args.model_name],
            "pretrained": [args.pretrained],
            "truncated_layers": [args.truncated_layers],
            "epochs": [args.epochs],
            "run_time": [run_time / 60],
            "lr": [args.lr],
            "image_size": [image_size],
            "rotate_angle": [args.rotate_angle],
            "horizontal_flip_prob": [args.horizontal_flip_prob],
            "gaussian_blur": [args.gaussian_blur],
            "normalize": [args.normalize],
            "seed": [args.seed],
            "gflops": [gflops],
            "params": [params]
        })

        new_results_df = pd.concat([new_results_df, boot_strap_results], axis=1)
        test_results_df = pd.concat([test_results_df, new_results_df], ignore_index=True)
        test_results_df.to_csv(f"{args.results_folder_name}/test_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a truncated EfficientNet model.")
    parser.add_argument("--data_dir", type=str, help="Directory containing the dataset.")
    parser.add_argument("--model_name", type=str, default="b0", choices=["b0", "b1", "b2", "b3"], help="EfficientNet model variant.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--rotate_angle", type=float, default=None, help="Rotation angle for data augmentation.")
    parser.add_argument("--horizontal_flip_prob", type=float, default=None, help="Probability of horizontal flip for data augmentation.")
    parser.add_argument("--gaussian_blur", type=float, default=None, help="Gaussian blur for data augmentation.")
    parser.add_argument("--normalize", action="store_true", help="Normalize the data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--truncated_layers", default=0, help="Number of layers to truncate from EfficientNet.")
    parser.add_argument("--bootstrap_n", type=int, default=200, help="Number of bootstrap iterations.")
    parser.add_argument("--results_folder_name", type=str, help="Folder to save results.")
    parser.add_argument("--save_logs", action="store_true", help="Save logs and outputs.")

    args = parser.parse_args()
    main(args)