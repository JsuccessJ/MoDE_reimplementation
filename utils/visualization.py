import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

def plot_loss_curves(history, figsize=(12, 5)):
    """
    Visualize training and validation loss curves.
    
    Args:
        history (dict): Training history containing 'train_loss', 'train_ce_loss', 'train_dist_loss', 'val_loss', 'val_acc'
        figsize (tuple): Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot training loss
    ax1.plot(history['train_loss'], label='Train Loss')
    if 'train_ce_loss' in history:
        ax1.plot(history['train_ce_loss'], label='CE Loss')
    if 'train_dist_loss' in history:
        ax1.plot(history['train_dist_loss'], label='Distillation Loss')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(history['val_loss'], label='Validation Loss')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    if 'val_acc' in history and history['val_acc']:
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_accuracy(moe_acc, mode_acc, labels=None, figsize=(10, 6)):
    """
    Compare MoE and MoDE model accuracies.
    
    Args:
        moe_acc (list): MoE model accuracies
        mode_acc (list): MoDE model accuracies
        labels (list): Experiment labels
        figsize (tuple): Figure size
    """
    if labels is None:
        labels = [f'Exp {i+1}' for i in range(len(moe_acc))]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width/2, moe_acc, width, label='MoE')
    rects2 = ax.bar(x + width/2, mode_acc, width, label='MoDE')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('MoE vs MoDE Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add values above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.show()

def visualize_expert_weights(model, inputs, cmap='viridis', figsize=(12, 5)):
    """
    Visualize the weights of each expert for a given input.
    
    Args:
        model: MoE or MoDE model
        inputs (torch.Tensor): Input data
        cmap (str): Color map
        figsize (tuple): Figure size
    """
    # Calculate gating network output
    if hasattr(model, 'moe'):  # For MoDE model
        gates = model.moe.get_gate_values(inputs)
    else:  # For MoE model
        gates = model.get_gate_values(inputs)
    
    # Average over batch dimension
    avg_gates = gates.mean(dim=0).detach().cpu().numpy()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    ax1.bar(range(len(avg_gates)), avg_gates)
    ax1.set_xlabel('Expert Index')
    ax1.set_ylabel('Average Weight')
    ax1.set_title('Average Expert Weights')
    ax1.set_xticks(range(len(avg_gates)))
    ax1.grid(True, alpha=0.3)
    
    # Heatmap (gating weights per sample)
    im = ax2.imshow(gates.detach().cpu().numpy()[:min(100, gates.shape[0])], 
                   aspect='auto', cmap=cmap)
    ax2.set_xlabel('Expert Index')
    ax2.set_ylabel('Sample Index (first 100)')
    ax2.set_title('Expert Weights per Sample')
    ax2.set_xticks(range(gates.shape[1]))
    fig.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    return avg_gates

def visualize_mnist_samples(train_loader, num_samples=10, save_path=None):
    """
    Visualize random MNIST samples from the dataset.
    
    Args:
        train_loader: DataLoader containing MNIST samples
        num_samples (int): Number of samples to visualize
        save_path (str, optional): Path to save the figure
    """
    # Get random samples
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Select a subset of images
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Create a grid of images
    img_grid = make_grid(images, nrow=5, normalize=True)
    
    # Convert to numpy and transpose
    img_grid = img_grid.numpy().transpose((1, 2, 0))
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.imshow(img_grid)
    plt.title('MNIST Samples')
    plt.axis('off')
    
    # Add labels below images
    for i, label in enumerate(labels):
        col = i % 5
        row = i // 5
        plt.text(col * 28 + 14, (row + 1) * 28 + 10, f'Label: {label.item()}', 
                 horizontalalignment='center', color='black')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_mnist_digits(train_loader, save_path=None):
    """
    Visualize samples of each digit (0-9) from MNIST dataset.
    
    Args:
        train_loader: DataLoader containing MNIST samples
        save_path (str, optional): Path to save the figure
    """
    # Get the entire batch
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Initialize array to store one example of each digit
    digit_images = [None] * 10
    
    # Find one example of each digit
    for i, label in enumerate(labels):
        digit = label.item()
        if digit_images[digit] is None:
            digit_images[digit] = images[i]
        
        # Break if we have found all digits
        if all(img is not None for img in digit_images):
            break
    
    # Create a new figure
    plt.figure(figsize=(12, 3))
    
    # Plot each digit
    for i, img in enumerate(digit_images):
        plt.subplot(1, 10, i+1)
        img = img.numpy().reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f'Digit: {i}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_expert_weights(weights, experts, save_path=None):
    """
    Plot the weights assigned to each expert by the gating network.
    
    Args:
        weights (torch.Tensor): Weights assigned to experts (batch_size x num_experts)
        experts (list): List of expert names or indices
        save_path (str, optional): Path to save the figure
    """
    # Convert to numpy if tensor
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    # Calculate average weight for each expert
    avg_weights = weights.mean(axis=0)
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(experts, avg_weights)
    plt.xlabel('Expert')
    plt.ylabel('Average Weight')
    plt.title('Average Expert Weights Assigned by Gating Network')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_training_metrics(metrics, save_path=None):
    """
    Plot training metrics (loss, accuracy) over epochs.
    
    Args:
        metrics (dict): Dictionary containing metrics
            Keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
            Values: List of values over epochs
        save_path (str, optional): Path to save the figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    ax1.plot(metrics['train_loss'], label='Training Loss')
    if 'val_loss' in metrics:
        ax1.plot(metrics['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot training and validation accuracy
    ax2.plot(metrics['train_acc'], label='Training Accuracy')
    if 'val_acc' in metrics:
        ax2.plot(metrics['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close() 