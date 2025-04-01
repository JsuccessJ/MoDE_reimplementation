import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from .moe import MoE, Expert, GatingNetwork

class MoDE(nn.Module):
    """
    Mixture of Distilled Experts (MoDE) model
    A model that applies mutual knowledge distillation based on MoE
    """
    def __init__(self, num_experts, input_dim, hidden_dim, output_dim, 
                 sparse_gate=False, top_k=None, distillation_temp=2.0, alpha=0.5):
        super(MoDE, self).__init__()
        
        # Create base MoE model
        self.moe = MoE(num_experts, input_dim, hidden_dim, output_dim, sparse_gate, top_k)
        
        # Hyperparameters for knowledge distillation
        self.distillation_temp = distillation_temp  # Distillation temperature
        self.alpha = alpha  # Parameter to adjust the weight between regular loss and distillation loss
        
        # Loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None
        
    def forward(self, x):
        """Forward propagation"""
        return self.moe(x)
    
    def get_expert_outputs(self, x):
        """Calculate outputs of each expert model individually"""
        expert_outputs = []
        for expert in self.moe.experts:
            expert_outputs.append(expert(x))
        return expert_outputs
        
    def configure_optimizer(self, lr=0.001):
        """Configure optimization algorithm"""
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def compute_distillation_loss(self, expert_outputs, gates):
        """
        Calculate distillation loss between experts
        Each expert learns from the outputs of other experts
        
        According to the paper:
        - Use Mean Squared Error (MSE) when the number of experts is 2
        - Use KL divergence when the number of experts is 3 or more
        """
        num_experts = len(expert_outputs)
        distillation_loss = 0
        
        # When the number of experts is 2: Use Mean Squared Error (MSE)
        if num_experts == 2:
            # Calculate MSE between outputs of the first and second experts
            # Apply softmax temperature
            softmax_output1 = F.softmax(expert_outputs[0] / self.distillation_temp, dim=1)
            softmax_output2 = F.softmax(expert_outputs[1] / self.distillation_temp, dim=1)
            
            # Calculate Mean Squared Error
            mse = F.mse_loss(softmax_output1, softmax_output2)
            
            # Apply gating weights
            weighted_mse = mse * gates[:, 0].mean() * gates[:, 1].mean()
            return weighted_mse
        
        # When the number of experts is 3 or more: Use KL divergence
        else:
            # Calculate KL divergence for each pair of experts
            for i in range(num_experts):
                for j in range(num_experts):
                    if i != j:  # Only between different experts
                        # Generate soft targets (convert logits to softmax probabilities)
                        soft_target = F.softmax(expert_outputs[j] / self.distillation_temp, dim=1)
                        # Student model's log softmax
                        log_pred = F.log_softmax(expert_outputs[i] / self.distillation_temp, dim=1)
                        
                        # Calculate KL divergence (between soft targets and predictions)
                        kl_div = F.kl_div(log_pred, soft_target, reduction='batchmean')
                        
                        # Apply gating weights to KL divergence
                        weighted_kl = kl_div * gates[:, i].mean() * gates[:, j].mean()
                        distillation_loss += weighted_kl
            
            # Normalize by the number of expert pairs
            return distillation_loss / (num_experts * (num_experts - 1))
    
    def train_step(self, x, y):
        """Single training step (including knowledge distillation)"""
        self.optimizer.zero_grad()
        
        # Calculate gating values and expert outputs
        gates = self.moe.get_gate_values(x)
        expert_outputs = self.get_expert_outputs(x)
        
        # Final output of MoE
        moe_output = self.forward(x)
        
        # Regular classification loss
        ce_loss = self.loss_fn(moe_output, y)
        
        # Knowledge distillation loss
        distillation_loss = self.compute_distillation_loss(expert_outputs, gates)
        
        # Modified to match the paper's formula: L = L_task + αL_KD
        total_loss = ce_loss + self.alpha * distillation_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'distillation_loss': distillation_loss.item()
        }
    
    def train_model(self, train_loader, val_loader=None, epochs=10, lr=0.001):
        """Train the entire model"""
        if self.optimizer is None:
            self.configure_optimizer(lr)
            
        self.train()
        history = {
            'train_loss': [], 
            'train_ce_loss': [], 
            'train_dist_loss': [],
            'val_loss': [], 
            'val_acc': []
        }
        
        for epoch in range(epochs):
            train_total_loss = 0
            train_ce_loss = 0
            train_dist_loss = 0
            
            for x_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                losses = self.train_step(x_batch, y_batch)
                train_total_loss += losses['total_loss']
                train_ce_loss += losses['ce_loss']
                train_dist_loss += losses['distillation_loss']
                
            # Average loss per epoch
            avg_train_loss = train_total_loss / len(train_loader)
            avg_ce_loss = train_ce_loss / len(train_loader)
            avg_dist_loss = train_dist_loss / len(train_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['train_ce_loss'].append(avg_ce_loss)
            history['train_dist_loss'].append(avg_dist_loss)
            
            # Perform validation (if available)
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, '
                      f'CE Loss: {avg_ce_loss:.4f}, Dist Loss: {avg_dist_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, '
                      f'CE Loss: {avg_ce_loss:.4f}, Dist Loss: {avg_dist_loss:.4f}')
                
        return history
    
    def evaluate(self, data_loader):
        """Evaluate on validation or test data"""
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                outputs = self(x_batch)
                loss = self.loss_fn(outputs, y_batch)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
        self.train()
        return total_loss / len(data_loader), correct / total
    
    def predict(self, x):
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
        return predicted 