import torch
import sys
import os

# Add root directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MoE, MoDE
from utils import load_mnist, plot_loss_curves, plot_accuracy, visualize_expert_weights

def main():
    # Load MNIST dataset
    train_loader, val_loader, test_loader = load_mnist(batch_size=64, val_ratio=0.1)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define MoE model
    moe_model = MoE(
        num_experts=2,
        input_dim=28*28,
        hidden_dim=128,
        output_dim=10
    ).to(device)
    
    # Define MoDE model
    mode_model = MoDE(
        num_experts=8,
        input_dim=28*28,
        hidden_dim=128,
        output_dim=10,
        distillation_temp=0.02,
        alpha=0.01
    ).to(device)
    
    # Train models
    # print("\nTraining MoE model...")
    # moe_history = train_model(moe_model, train_loader, val_loader, device, epochs=5, lr=0.001)
    
    print("\nTraining MoDE model...")
    mode_history = train_model(mode_model, train_loader, val_loader, device, epochs=5, lr=0.001)
    
    # # Evaluate on test data
    # print("\nEvaluating models on test data...")
    # moe_test_loss, moe_test_acc = evaluate_model(moe_model, test_loader, device)
    mode_test_loss, mode_test_acc = evaluate_model(mode_model, test_loader, device)
    
    # print(f"MoE Test Accuracy: {moe_test_acc:.4f}")
    print(f"MoDE Test Accuracy: {mode_test_acc:.4f}")
    
    # # Visualize performance comparison
    # plot_accuracy(
    #     [moe_test_acc], 
    #     [mode_test_acc], 
    #     labels=['MNIST']
    # )
    
    # # Visualize loss curves
    # print("\nPlotting loss curves...")
    # plot_loss_curves(moe_history)
    # plot_loss_curves(mode_history)
    
    # # Visualize expert weights
    # print("\nVisualizing expert weights...")
    # # Sample a batch from test data
    # for x_batch, _ in test_loader:
    #     x_batch = x_batch.view(x_batch.size(0), -1).to(device)
    #     break
    
    # visualize_expert_weights(moe_model, x_batch)
    # visualize_expert_weights(mode_model, x_batch)

def train_model(model, train_loader, val_loader, device, epochs=5, lr=0.001):
    """Model training function"""
    # Function to move data to device
    def to_device(data, target):
        data = data.view(data.size(0), -1).to(device)  # Convert MNIST images to vectors
        target = target.to(device)
        return data, target
    
    # Training loop
    if hasattr(model, 'train_model'):
        # Use training method from MoDE or MoE class
        history = {}
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = to_device(x_batch, y_batch)
            sample_batch = (x_batch, y_batch)
            break
        
        # Wrap data loaders with transformed format
        class DeviceLoader:
            def __init__(self, loader, device):
                self.loader = loader
                self.device = device
            
            def __iter__(self):
                for data, target in self.loader:
                    yield to_device(data, target)
            
            def __len__(self):
                return len(self.loader)
        
        train_dev_loader = DeviceLoader(train_loader, device)
        val_dev_loader = DeviceLoader(val_loader, device) if val_loader else None
        
        # Call model training method
        history = model.train_model(train_dev_loader, val_dev_loader, epochs=epochs, lr=lr)
    
    return history

def evaluate_model(model, test_loader, device):
    """Model evaluation function"""
    # Move data to device
    def to_device(data, target):
        data = data.view(data.size(0), -1).to(device)  # Convert MNIST images to vectors
        target = target.to(device)
        return data, target
    
    # Wrap evaluation loader
    class DeviceLoader:
        def __init__(self, loader, device):
            self.loader = loader
            self.device = device
        
        def __iter__(self):
            for data, target in self.loader:
                yield to_device(data, target)
        
        def __len__(self):
            return len(self.loader)
    
    test_dev_loader = DeviceLoader(test_loader, device)
    
    # Evaluate model
    loss, acc = model.evaluate(test_dev_loader)
    return loss, acc

if __name__ == "__main__":
    main() 