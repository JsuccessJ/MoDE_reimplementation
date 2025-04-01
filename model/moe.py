import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class Expert(nn.Module):
    """Base network for expert models"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class GatingNetwork(nn.Module):
    """Gating network that assigns weights to each expert model"""
    def __init__(self, input_dim, num_experts, hidden_dim=64):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class MoE(nn.Module):
    """Mixture of Experts model"""
    def __init__(self, num_experts, input_dim, hidden_dim, output_dim, sparse_gate=False, top_k=None):
        super(MoE, self).__init__()
        
        self.num_experts = num_experts
        self.sparse_gate = sparse_gate
        self.top_k = top_k if top_k is not None else num_experts
        
        # Create expert models
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) 
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = GatingNetwork(input_dim, num_experts)
        
        # Attributes for optimization
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None
        
    def forward(self, x):
        # Calculate gating weights for each expert
        gates = self.gate(x)
        
        # Apply sparse gating (optional)
        if self.sparse_gate and self.top_k < self.num_experts:
            # Select only top-k experts
            top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=1)
            # Calculate normalized weights
            top_k_gates = top_k_gates / top_k_gates.sum(dim=1, keepdim=True)
            
            # Initialize expert outputs
            expert_outputs = torch.zeros(x.size(0), self.experts[0].fc3.out_features, device=x.device)
            
            # Weighted sum of outputs from selected experts
            for i, expert in enumerate(self.experts):
                # Find samples where i-th expert is selected
                mask = (top_k_indices == i).any(dim=1)
                if mask.any():
                    # Calculate expert output only for selected samples
                    expert_output = expert(x[mask])
                    # Apply gating weights
                    expert_outputs[mask] += expert_output * gates[mask, i].unsqueeze(1)
        else:
            # Weighted sum of outputs from all experts
            expert_outputs = torch.zeros(x.size(0), self.experts[0].fc3.out_features, device=x.device)
            for i, expert in enumerate(self.experts):
                expert_output = expert(x)
                expert_outputs += expert_output * gates[:, i].unsqueeze(1)
                
        return expert_outputs
    
    def get_gate_values(self, x):
        """Return gating network values for input"""
        return self.gate(x)
    
    def configure_optimizer(self, lr=0.001):
        """Configure optimization algorithm"""
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def train_step(self, x, y):
        """Single training step"""
        self.optimizer.zero_grad()
        outputs = self(x)
        loss = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train_model(self, train_loader, val_loader=None, epochs=10, lr=0.001):
        """Train the entire model"""
        if self.optimizer is None:
            self.configure_optimizer(lr)
            
        self.train()
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            train_loss = 0
            for x_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                batch_loss = self.train_step(x_batch, y_batch)
                train_loss += batch_loss
                
            # Average loss per epoch
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Perform validation (if available)
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}')
                
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