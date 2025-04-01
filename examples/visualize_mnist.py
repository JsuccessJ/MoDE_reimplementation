import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add root directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_mnist

def visualize_mnist_samples(num_samples=25):
    """Visualize random samples from the MNIST dataset"""
    
    # Load MNIST dataset
    train_loader, _, _ = load_mnist(batch_size=100, val_ratio=0)
    
    # Get one batch of data
    for images, labels in train_loader:
        break
    
    # Check number of samples to display
    num_samples = min(num_samples, len(images))
    
    # Calculate grid size (e.g., 5x5)
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Visualization
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(images[i][0], cmap='gray')  # MNIST is single channel (grayscale)
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png')  # Save image
    plt.show()

def visualize_specific_digits():
    """Visualize example images for each digit (0-9)"""
    
    # Load MNIST dataset
    train_loader, _, _ = load_mnist(batch_size=10000, val_ratio=0)
    
    # Get all training data from the first batch
    for images, labels in train_loader:
        break
    
    # Find images for each digit
    digit_indices = {}
    for i in range(10):
        # Find the first index for each digit
        indices = (labels == i).nonzero()
        if len(indices) > 0:
            digit_indices[i] = indices[0].item()
    
    # Visualization
    plt.figure(figsize=(12, 2.5))
    for i in range(10):
        if i in digit_indices:
            plt.subplot(1, 10, i+1)
            plt.imshow(images[digit_indices[i]][0], cmap='gray')
            plt.title(f'Digit: {i}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_digits_0_to_9.png')  # Save image
    plt.show()

if __name__ == "__main__":
    print("Visualizing MNIST dataset samples")
    visualize_mnist_samples(25)
    print("Example images for each digit (0-9)")
    visualize_specific_digits() 