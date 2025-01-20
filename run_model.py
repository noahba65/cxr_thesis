#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch import nn
from poutyne import Model, CSVLogger, ModelCheckpoint, EarlyStopping, plot_history
import numpy as np
import torchmetrics
from datetime import datetime
import pandas as pd
from custom_lib.custom_model import NeuralNetwork
from custom_lib.data_prep import data_transformation_pipeline, data_loader
from custom_lib.utils import load_pretrained_model
import time
import argparse

def arg_parser():
    """
    Parses command-line arguments for image classification tasks.

    Returns:
        Namespace: Parsed command-line arguments as attributes of an argparse Namespace object.

    Arguments:
        --model_name (str): Model name from torch.models (default: 'efficientnet_b0').
        --epochs (int): Number of epochs for training (default: 15).
        --save_logs (bool): Save logs to the results folder if set to True (default: True).
        --lr (float): Learning rate for training (default: 1e-3).
        --image_size (int): Image size in pixels (both width and height, default: 224).
        --rotate_angle (float): Maximum rotation angle for image augmentation (default: 0).
        --horizontal_flip_prob (float): Probability of horizontal flip for augmentation (default: 0.5).
        --gaussian_blur (float): Gaussian blur kernel size for augmentation (default: None).
        --normalize (bool): Normalize image pixel values if set to True (default: False).
    """
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Image classification')

    # Model and training-related arguments
    parser.add_argument('--model_name', default='efficientnet_b0', help='Model name from torch.models. More info at https://pytorch.org/vision/0.9/models.html')
    parser.add_argument('--epochs', default=15, type=int, help='Number of epochs for training.')
    parser.add_argument('--save_logs', type=bool, default=True, help='Save logs to the results folder if set to True.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training.')
    parser.add_argument('--image_size', type=int, default=224, help='Image size in pixels (width and height).')

    # Data augmentation-related arguments
    parser.add_argument('--rotate_angle', type=float, default=0, help='Maximum rotation angle for image augmentation.')
    parser.add_argument('--horizontal_flip_prob', type=float, default=None, help='Probability of horizontal flip for augmentation.')
    parser.add_argument('--gaussian_blur', type=float, default=None, help='Gaussian blur kernel size for augmentation. Leave as None for no blur.')

    # Normalization argument
    parser.add_argument('--normalize', type=bool, default=False, help='Normalize image pixel values if set to True.')

    # Parse arguments
    args = parser.parse_args()

    return args

def train_and_evaluate(
    data_dir,
    model_name,
    save_logs,
    epochs,
    lr,
    image_size,
    rotate_angle=None,
    horizontal_flip_prob=None,
    gaussian_blur=None,
    normalize=False
):
    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Prepare data transformations and loaders
    train_transform = data_transformation_pipeline(
        image_size=image_size,
        rotate_angle=rotate_angle,
        horizontal_flip_prob=horizontal_flip_prob,
        gaussian_blur=gaussian_blur,
        normalize=normalize,
        is_train=True,
    )
    test_transform = data_transformation_pipeline(
        image_size=image_size,
        rotate_angle=rotate_angle,
        horizontal_flip_prob=horizontal_flip_prob,
        gaussian_blur=gaussian_blur,
        normalize=normalize,
        is_train=False,
    )
    val_transform = data_transformation_pipeline(
        image_size=image_size,
        rotate_angle=rotate_angle,
        horizontal_flip_prob=horizontal_flip_prob,
        gaussian_blur=gaussian_blur,
        normalize=normalize,
        is_train=False,
    )

    train_loader, val_loader, test_loader, num_classes = data_loader(
        data_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        val_transform=val_transform,
    )

    model = load_pretrained_model(model_name)

    # Wrap the model with Poutyne
    poutyne_model = Model(
        model,
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        loss_function=nn.CrossEntropyLoss(),
        batch_metrics=["accuracy"],
        device=device,
    )

    # Set up logging
    if save_logs:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        # Create directory for saving all logs and model outputs 
        results_dir = os.path.join(f"results/{model_name}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        print(f"Logs and output will be saved in: {results_dir}")

        # Callbacks for logging and early stopping
        checkpoint = ModelCheckpoint(
            f"{results_dir}/best_model.pth", monitor="val_loss", mode="min", save_best_only=True
        )
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        csv_logger = CSVLogger(f"{results_dir}/training_logs.csv")
        callbacks = [checkpoint, early_stopping, csv_logger]
    else:
        callbacks = None

    # Train the model
    start_time = time.time()
    print("Training Model")
    history = poutyne_model.fit_generator(
        train_loader, val_loader, epochs=epochs, verbose=True, callbacks=callbacks
    )
    end_time = time.time()
    run_time = end_time - start_time
    print(f"Model training took {run_time / 60:.2f} minutes")

    # Evaluate the model
    test_metrics = poutyne_model.evaluate_generator(test_loader)
    print("Test metrics:", test_metrics)

    # Save logs and plots
    if save_logs:
        with open(f"{results_dir}/model_overview.txt", "w") as file:
            file.write(f"Test Loss: {test_metrics[0]}\n")
            file.write(f"Test Acc: {test_metrics[1]}\n")
            file.write(f"Model Structure:\n{model}\n")
            file.write(f"Using {device} device\n")

 # Check if the CSV file exists
        if os.path.exists("results/test_results.csv"):
            # Read the existing CSV file into a DataFrame
            test_results_df = pd.read_csv("results/test_results.csv")
        else:
            # If the file doesn't exist, create an empty DataFrame or initialize with test output columns
            test_results_df = pd.DataFrame(columns=[
            "model_id", "model", "epochs", "run_time", "test_loss", "test_acc", "lr", 
            "image_size", "rotate_angle", "horizontal_flip_prob", 
            "gaussian_blur", "normalize"
            ])

        # Create a dictionary for the new results
        new_results_dict = {
            "model_id": [f"{model_name}_{timestamp}"],
            "model": [model_name],
            "epochs": [epochs],  
            "run_time": [run_time],  
            "test_loss": [test_metrics[0]],
            "test_acc": [test_metrics[1]],
            "lr": [lr],
            "image_size": [image_size],  
            "rotate_angle": [rotate_angle],  
            "horizontal_flip_prob": [horizontal_flip_prob],  
            "gaussian_blur": [gaussian_blur],  
            "normalize": [normalize]  
        }


        new_rows_df = pd.DataFrame(new_results_dict)

        test_results_df = pd.concat([test_results_df, new_rows_df], ignore_index=True).fillna(value='None')


        test_results_df.to_csv("results/test_results.csv", index=False)


    return history


if __name__ == "__main__":
    # Parse args from commandline
    args = arg_parser()

    ################    Train and Evaluate Model    ########################
    train_and_evaluate(
        data_dir="data", 
        model_name=args.model_name,
        save_logs=args.save_logs,
        epochs=args.epochs,
        lr=args.lr,
        image_size=args.image_size,
        rotate_angle=args.rotate_angle,
        horizontal_flip_prob=args.horizontal_flip_prob,
        gaussian_blur=args.gaussian_blur,
        normalize=args.normalize,
    )

