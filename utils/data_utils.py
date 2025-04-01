import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_mnist(batch_size=128, val_ratio=0.1, random_seed=42):
    """
    Load MNIST dataset and return train/validation/test data loaders
    
    Args:
        batch_size (int): Batch size
        val_ratio (float): Validation data ratio
        random_seed (int): Random seed
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load training data
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Split training/validation
    if val_ratio > 0:
        val_size = int(len(train_dataset) * val_ratio)
        train_size = len(train_dataset) - val_size
        
        torch.manual_seed(random_seed)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Load test data
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def load_cifar10(batch_size=128, val_ratio=0.1, random_seed=42):
    """
    Load CIFAR-10 dataset and return train/validation/test data loaders
    
    Args:
        batch_size (int): Batch size
        val_ratio (float): Validation data ratio
        random_seed (int): Random seed
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Data transformation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    # Split training/validation
    if val_ratio > 0:
        val_size = int(len(train_dataset) * val_ratio)
        train_size = len(train_dataset) - val_size
        
        torch.manual_seed(random_seed)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Apply test transformation to validation data
        val_dataset.dataset.transform = transform_test
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def load_custom_dataset(X, y, batch_size=128, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Load custom dataset and return train/validation/test data loaders
    
    Args:
        X (numpy.ndarray): Input data
        y (numpy.ndarray): Target data
        batch_size (int): Batch size
        val_ratio (float): Validation data ratio
        test_ratio (float): Test data ratio
        random_seed (int): Random seed
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Convert NumPy arrays to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split training/validation/test
    test_size = int(len(dataset) * test_ratio)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - test_size - val_size
    
    torch.manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader 