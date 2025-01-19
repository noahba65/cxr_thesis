import torchvision.models as models
from argparse import Namespace
import argparse
import configparser

def load_pretrained_model(model_name):
    """
    Loads a pretrained model from torchvision.models.

    Args:
        model_name (str): The name of the model to load. This should be one of the available models in 
                          torchvision.models (e.g., 'resnet18', 'efficientnet_b0', etc.).

    Raises:
        ValueError: If the specified model is not available in torchvision.models.

    Returns:
        torch.nn.Module: The loaded model with pretrained weights.
    """
    # Initialize model
    model_class = getattr(models, model_name, None)
    if model_class is None:
        raise ValueError(f"Model '{model_name}' is not available in torchvision.models.")
    
    # Load the model with pretrained weights
    model = model_class(pretrained=True)

    return model


def parse_args_with_config():
    """
    Parses command-line arguments and optionally reads values from a config file.

    Returns:
        Namespace: A Namespace object containing the final values of all arguments.
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Parse command-line arguments with optional config file.")
    
    # Add command-line arguments
    parser.add_argument("--config_file", type=str, help="Path to the config file")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the data directory")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--save_logs", type=bool, default=True, help="Whether to save logs")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=224, help="Image size for data transformations")
    parser.add_argument("--rotate_angle", type=float, default=None, help="Rotation angle for augmentation")
    parser.add_argument("--horizontal_flip_prob", type=float, default=None, help="Horizontal flip probability")
    parser.add_argument("--brightness_contrast", type=float, default=None, help="Brightness and contrast adjustment")
    parser.add_argument("--gaussian_blur", type=float, default=None, help="Gaussian blur level")
    parser.add_argument("--normalize", type=bool, default=False, help="Whether to normalize data")

    # Parse arguments
    args = parser.parse_args()

    # Convert args to a dictionary for easier processing
    args_dict = vars(args)

    # If a config file is specified, load values from it
    if args.config_file:
        config = configparser.ConfigParser()
        config.read(args.config_file)
        settings = config["settings"]

        # Update args_dict with values from the config file if not already provided
        for key in settings:
            # Set the value if not already provided via command line
            if key in args_dict and args_dict[key] is None:
                args_dict[key] = settings.get(key)

        # Convert types for certain arguments if necessary (e.g., boolean or numeric)
        args_dict["save_logs"] = args_dict["save_logs"] == "True" if isinstance(args_dict["save_logs"], str) else args_dict["save_logs"]
        args_dict["rotate_angle"] = float(args_dict["rotate_angle"]) if args_dict["rotate_angle"] is not None else None
        args_dict["horizontal_flip_prob"] = float(args_dict["horizontal_flip_prob"]) if args_dict["horizontal_flip_prob"] is not None else None
        args_dict["brightness_contrast"] = float(args_dict["brightness_contrast"]) if args_dict["brightness_contrast"] is not None else None
        args_dict["gaussian_blur"] = float(args_dict["gaussian_blur"]) if args_dict["gaussian_blur"] is not None else None

    return Namespace(**args_dict)  # Convert dictionary back to Namespace


