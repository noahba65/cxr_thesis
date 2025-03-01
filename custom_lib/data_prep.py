import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, Resize, CenterCrop, ToTensor, Normalize, GaussianBlur, ColorJitter
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset

def data_transformation_pipeline(
    image_size,
    center_crop,
    rotate_angle,
    horizontal_flip_prob,
    gaussian_blur,
    normalize,
    is_train=False
):
    """
    Dynamically creates a data transformation pipeline.

    Args:
        image_size (int): Target size for resizing and cropping.
        rotate_angle (int, optional): Maximum rotation angle in degrees. Default is None (no rotation).
        horizontal_flip_prob (float, optional): Probability of horizontal flip. Default is None (no flipping).
        gaussian_blur (int, optional): Kernel size for Gaussian blur. Default is None (no blur).
        normalize (bool): Whether to include normalization. Default is False.
        is_train (bool): If True, include training-specific transformations. Default is False.

    Returns:
        torchvision.transforms.Compose: A composed transformation pipeline.
    """
    transform_steps = []

    # Basic resizing and cropping
    transform_steps.append(Resize(image_size))
    transform_steps.append(CenterCrop(center_crop))

    # Training-specific augmentations
    if is_train:
        if rotate_angle is not None:
            transform_steps.append(RandomRotation(degrees=rotate_angle))
        if horizontal_flip_prob is not None:
            transform_steps.append(RandomHorizontalFlip(p=horizontal_flip_prob))
        if gaussian_blur is not None:
            transform_steps.append(GaussianBlur(kernel_size=gaussian_blur))

    # Convert to tensor
    transform_steps.append(ToTensor())

    # Normalization
    if normalize:
        transform_steps.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return Compose(transform_steps)


def data_loader(data_dir, train_transform, val_transform, test_transform, seed, 
                batch_size, train_prop, val_prop, num_workers=4):
    """
    Creates train, validation, and test DataLoaders from a dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        train_transform (torchvision.transforms.Compose): Transformations to apply to training images.
        val_transform (torchvision.transforms.Compose): Transformations to apply to validation images.
        test_transform (torchvision.transforms.Compose): Transformations to apply to test images.
        batch_size (int): Batch size for the DataLoader.
        train_prop (float): Proportion of the dataset used for training.
        val_prop (float): Proportion of the dataset used for validation.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: Train, validation, and test DataLoaders, and the number of classes.
    """

    # Validate proportions
    assert train_prop + val_prop < 1, "The sum of train_prop and val_prop must be less than 1."

    # Load the dataset with a placeholder transform
    data = ImageFolder(data_dir, transform=None)
    data_length = len(data)

    # Number of classes
    num_classes = len(data.classes)

    # Define the sizes for the splits
    train_size = int(train_prop * data_length)
    val_size = int(val_prop * data_length)
    test_size = data_length - train_size - val_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("Dataset is too small for the specified split proportions.")

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Split the dataset
    trainset, valset, testset = random_split(data, [train_size, val_size, test_size])

    # Apply specific transformations
    trainset.dataset.transform = train_transform
    valset.dataset.transform = val_transform
    testset.dataset.transform = test_transform

    # Create DataLoaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(valset, batch_size=batch_size * 2, num_workers=num_workers, pin_memory=True,
                            drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size * 2, num_workers=num_workers, pin_memory=True,
                             drop_last=True)

    # Log split sizes
    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

    return train_loader, val_loader, test_loader, num_classes
