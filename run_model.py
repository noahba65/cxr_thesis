import argparse
import os
import torch
from torch import nn
from poutyne import Model, CSVLogger
from poutyne.framework import ModelCheckpoint, EarlyStopping, plot_history
import numpy as np
import torchmetrics
from datetime import datetime
import sys
import pandas as pd
from custom_lib.custom_models.basic_nn import NeuralNetwork
from custom_lib.data_prep import data_transformation_pipeline, data_loader
import matplotlib as plt
import torchvision.models as models
import time
import importlib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from poutyne import ReduceLROnPlateau, Callback
from thop import profile


def load_model(model_name, **kwargs):
    """Dynamically loads and instantiates a model from custom_lib.custom_models."""
    module = importlib.import_module(f"custom_lib.custom_models.{model_name}")
    
    # Find the first class in the module (assuming only one model class per file)
    model_class = getattr(module, model_name, None)
    
    if model_class is None:
        raise ValueError(f"Could not find a class named '{model_name}' in '{module.__name__}'")

    return model_class(**kwargs)


class PrintLRSchedulerCallback(Callback):
    def set_model(self, model):
        self.model = model  # Store the model reference

    def on_epoch_end(self, epoch, logs):
        lr = self.model.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}: Current LR = {lr}")

def compute_model_stats(model, batch_size, image_size, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Computes the FLOPs and number of parameters for a given model.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        batch_size (int): The batch size for the dummy input.
        image_size (int): The height and width of the input image.
        device (str): The device to perform computations on ("cuda" or "cpu").
    
    Returns:
        tuple: (GFLOPs, parameters)
    """
    model.to(device)  # Move model to the specified device
    dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)

    flops, params = profile(model, inputs=(dummy_input,))
    gflops = flops / 1e9  # Convert to GFLOPs

    print(f"GFLOPs: {gflops:.3f}")
    print(f"Parameters: {params:,}")  # Add commas for readability

    return gflops, params       

def main(args):
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
    print(f"Using {device} device")

    train_transform = data_transformation_pipeline(image_size = args.image_size,
                                                   center_crop=args.center_crop,
                                                   rotate_angle=args.rotate_angle,
                                                   horizontal_flip_prob=args.horizontal_flip_prob,
                                                   gaussian_blur=args.gaussian_blur,
                                                   normalize=args.normalize,
                                                   is_train=True)
    test_transform = data_transformation_pipeline(image_size = args.image_size,
                                                rotate_angle=args.rotate_angle,
                                                horizontal_flip_prob=args.horizontal_flip_prob,
                                                center_crop=args.center_crop,
                                                gaussian_blur=args.gaussian_blur,
                                                normalize=args.normalize,
                                                is_train=False)
    val_transform = data_transformation_pipeline(image_size = args.image_size,
                                                rotate_angle=args.rotate_angle,
                                                center_crop=args.center_crop,
                                                horizontal_flip_prob=args.horizontal_flip_prob,
                                                gaussian_blur=args.gaussian_blur,
                                                normalize=args.normalize,
                                                is_train=False)


    train_loader , val_loader, test_loader, num_classes = data_loader(args.data_dir, 
                                                        train_transform=train_transform,
                                                        test_transform=test_transform,
                                                        val_transform=val_transform,
                                                        seed=args.seed,
                                                        batch_size=args.batch_size,
                                                        train_prop=args.train_prop,
                                                        val_prop=args.val_prop
                                                        )
    
    model = load_model(
                args.model_name,
                num_classes=num_classes,
                removed_layers=args.truncated_layers,
                batch_size=args.batch_size,
                image_size=args.image_size,
                pretrained=args.pretrained,
                dropout_p=args.dropout_p
                        )
    
    if args.save_logs:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Create directory for saving all logs and model outputs 
        results_dir = os.path.join(f"{args.results_folder_name}/{args.model_name}_reduced_layers_{args.truncated_layers}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        print(f"Logs and output will be saved in: {results_dir}")
        
    poutyne_model = Model(
                        model,
                        # optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),  # Added momentum

                        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
                        loss_function=nn.CrossEntropyLoss(),
                        batch_metrics=["accuracy"],
                        device=device
                        )

    # Add the ReduceLROnPlateau callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # Monitor validation loss
        factor=0.1,          # Reduce LR by a factor of 0.1
        patience=5           # Wait 5 epochs before reducing LR

    )
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)



    # Instantiate the callback
    print_lr_callback = PrintLRSchedulerCallback()

    # Add it to the list of callbacks
    # callbacks = [reduce_lr, early_stopping, print_lr_callback]
    callbacks = [reduce_lr, print_lr_callback]

    if args.save_logs == True:
        # Callback: Save the best model based on validation accuracy
        checkpoint = ModelCheckpoint(f"{results_dir}/best_model.pth", monitor='val_loss', mode='min', save_best_only=True)
        csv_logger = CSVLogger(f"{results_dir}/training_logs.csv")
        callbacks = [checkpoint, csv_logger, reduce_lr, print_lr_callback]
        

    start_time = time.time()
    # 7. Train the model
    history = poutyne_model.fit_generator(train_loader, val_loader, epochs=args.epochs, verbose=True,
                                callbacks = callbacks)
    end_time = time.time()

    run_time = end_time - start_time

    print(f"Model training took {run_time / 60} minutes")

    if args.save_logs:
        # Save the final model manually
        torch.save(poutyne_model.network.state_dict(), f"{results_dir}/final_model.pth")

    if args.save_logs:
        best_model_path = f"{results_dir}/best_model.pth"
        
        # Load the state dict into the model
        poutyne_model.network.load_state_dict(torch.load(best_model_path))


    
    if args.bootstrap_n == None:

        print("Starting single test evalution")
        # Evaluate using Poutyne
        test_loss, test_acc = poutyne_model.evaluate_generator(test_loader)
    else: 
        from custom_lib.bootstrap import bootstrap_evaluation_poutyne

        if args.save_logs:

            print("Starting bootstrap evalution")

            # Run bootstrapping evaluation with your Poutyne model
            boot_strap_results = bootstrap_evaluation_poutyne(poutyne_model, test_loader, n_bootstraps = args.bootstrap_n, 
                                                          save_logs=args.save_logs, results_dir=results_dir, seed=args.seed)
        else:
            print("Starting bootstrap evalution")
            boot_strap_results = bootstrap_evaluation_poutyne(poutyne_model, test_loader, n_bootstraps = args.bootstrap_n,
                                                           save_logs=args.save_logs, seed=args.seed)

    gflops, params = compute_model_stats(model, batch_size=args.batch_size, image_size=args.image_size)
  
    # Save logs and plots
    if args.save_logs:
        with open(f"{results_dir}/model_overview.txt", "w") as file:
            file.write(f"Model Structure:\n{model}\n")
            file.write(f"Using {device} device\n")

        # Check if CSV exists
        if os.path.exists(f"{args.results_folder_name}/test_results.csv"):
            test_results_df = pd.read_csv(f"{args.results_folder_name}/test_results.csv")
        else:
            test_results_df = pd.DataFrame(columns=[
                "model_id", "model", "epochs", "run_time", "lr", "image_size",
                "rotate_angle", "horizontal_flip_prob", "gaussian_blur", "normalize", "seed", "truncated_layers"
            ])

        if args.bootstrap_n != None:
            test_loss = None
            test_acc = None


        # Create a DataFrame for the new model's metadata
        new_results_df = pd.DataFrame({
            "model_id": [f"{args.model_name}_reduced_layers_{args.truncated_layers}_{timestamp}"],
            "model": [args.model_name],
            "truncated_layers": [args.truncated_layers],
            "epochs": [args.epochs],  
            "batch_size": [args.batch_size],
            "run_time": [run_time / 60],  
            "lr": [args.lr],
            "image_size": [args.image_size],  
            "rotate_angle": [args.rotate_angle],  
            "horizontal_flip_prob": [args.horizontal_flip_prob],  
            "gaussian_blur": [args.gaussian_blur],  
            "normalize": [args.normalize],
            "seed": [args.seed],
            "gflops": [gflops],
            "params": [params],
            "single_test_acc": [test_acc],
            "single_test_loss": [test_loss],
            "bootstrap_n": [args.bootstrap_n],
            "train_prop": [args.train_prop],
            "val_prop": [args.val_prop]
            })

        if args.bootstrap_n != None:
            # Combine test metadata with bootstrapped results (column-wise merge)
            new_results_df = pd.concat([new_results_df, boot_strap_results], axis=1)


        # Append to existing DataFrame
        test_results_df = pd.concat([test_results_df, new_results_df], ignore_index=True)

        # Save updated results
        test_results_df.to_csv(f"{args.results_folder_name}/test_results.csv", index=False)

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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a truncated EfficientNet model.")
    parser.add_argument("--data_dir", type=str, help="Directory containing the dataset.")
    parser.add_argument("--model_name", type=str, choices=["truncated_b0", "truncated_b0_leaky", "truncated_b0_leaky2"], help="Custom model found in custom_lib.custom_models.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--rotate_angle", type=float, default=None, help="Rotation angle for data augmentation.")
    parser.add_argument("--horizontal_flip_prob", type=float, default=None, help="Probability of horizontal flip for data augmentation.")
    parser.add_argument("--gaussian_blur", type=float, default=None, help="Gaussian blur for data augmentation.")
    parser.add_argument("--normalize", action="store_true", help="Normalize the data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--truncated_layers", type=int, default=0, help="Number of layers to truncate from EfficientNet.")
    parser.add_argument("--bootstrap_n", type=int, default=None, help="Number of bootstrap iterations.")
    parser.add_argument("--results_folder_name", type=str, help="Folder to save results.")
    parser.add_argument("--save_logs", action="store_true", help="Save logs and outputs.")
    parser.add_argument("--image_size", type=int, default=224, help="Size of image for resize in data transform")
    parser.add_argument("--center_crop", type = int, default=224, help="Centercrop of image in data transform")
    parser.add_argument("--train_prop", type=float, default=.8, help="What proportion to split training data on.")
    parser.add_argument("--val_prop", type=float, default=.1, help="Proportion for validation set.")
    parser.add_argument("--dropout_p", type=float, default=.2, help="The probablity for the dropout in classifier layer.")

    args = parser.parse_args()
    main(args)