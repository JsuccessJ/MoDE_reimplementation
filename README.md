# MoDE (Mixture of Distilled Experts) Implementation

This project is an implementation of the [MoDE: A Mixture-of-Experts Model with Mutual Distillation among the Experts](https://arxiv.org/abs/2402.00893) model. MoDE is an improved version of MoE (Mixture of Experts) that uses expert models, enhancing the model's generalization ability through mutual knowledge distillation.

## Core Architecture

MoDE model consists of the following key components:

- **Expert Models**: Each expert is a neural network specialized in processing specific sub-tasks.
- **Gating Network**: Analyzes the input and assigns weights to each expert.
- **Mutual Distillation**: A mechanism for sharing knowledge between experts.

In traditional MoE models, the gating network routes input features to different experts, allowing each to specialize in handling specific sub-tasks. However, this approach limits experts to learning from only a subset of samples, restricting their generalization ability. MoDE addresses this issue by introducing mutual distillation among experts. This enables each expert to leverage features learned by other experts, gaining more accurate perceptions of their originally allocated sub-tasks.

### Code Implementation

The core implementation of MoDE consists of the following components:

1. **Base MoE Implementation**:

```python
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
            
            # Weighted sum of outputs from selected experts
            expert_outputs = torch.zeros(x.size(0), self.experts[0].fc3.out_features, device=x.device)
            for i, expert in enumerate(self.experts):
                # Find samples where i-th expert is selected
                mask = (top_k_indices == i).any(dim=1)
                if mask.any():
                    expert_output = expert(x[mask])
                    expert_outputs[mask] += expert_output * gates[mask, i].unsqueeze(1)
        else:
            # Weighted sum of outputs from all experts
            expert_outputs = torch.zeros(x.size(0), self.experts[0].fc3.out_features, device=x.device)
            for i, expert in enumerate(self.experts):
                expert_output = expert(x)
                expert_outputs += expert_output * gates[:, i].unsqueeze(1)
                
        return expert_outputs
```

2. **MoDE Implementation with Mutual Distillation**:

```python
class MoDE(nn.Module):
    """Mixture of Distilled Experts (MoDE) model"""
    def __init__(self, num_experts, input_dim, hidden_dim, output_dim, distillation_temp=2.0, 
                 alpha=0.5, sparse_gate=False, top_k=None):
        super(MoDE, self).__init__()
        
        self.alpha = alpha  # Weight for distillation loss
        self.distillation_temp = distillation_temp  # Temperature for knowledge distillation
        
        # Initialize the MoE model
        self.moe = MoE(num_experts, input_dim, hidden_dim, output_dim, sparse_gate, top_k)
        
    def forward(self, x):
        return self.moe(x)
    
    def compute_distillation_loss(self, expert_outputs, gates):
        """Compute mutual distillation loss among experts"""
        distillation_loss = 0
        
        if self.training and self.alpha > 0:
            # Get the number of experts
            num_experts = len(self.moe.experts)
            
            # Compute distillation loss for each expert pair
            for i in range(num_experts):
                for j in range(num_experts):
                    if i != j:
                        # Get soft targets from expert j
                        with torch.no_grad():
                            soft_targets = F.softmax(expert_outputs[j] / self.distillation_temp, dim=1)
                            
                        # Get log probabilities from expert i
                        log_probs = F.log_softmax(expert_outputs[i] / self.distillation_temp, dim=1)
                        
                        # Compute KL divergence
                        kl_div = F.kl_div(log_probs, soft_targets, reduction='batchmean')
                        
                        # Weight by the average gate value for expert i
                        avg_gate = gates[:, i].mean()
                        distillation_loss += kl_div * avg_gate
                        
            # Normalize by the number of expert pairs
            distillation_loss /= num_experts * (num_experts - 1)
            
            # Scale by temperature squared (as in original KD paper)
            distillation_loss *= (self.distillation_temp ** 2)
        
        return distillation_loss
    
    def train_step(self, x, y):
        """Single training step with distillation"""
        self.moe.optimizer.zero_grad()
        
        # Forward pass
        outputs = self(x)
        
        # Calculate standard cross-entropy loss
        ce_loss = self.moe.loss_fn(outputs, y)
        
        # Calculate individual expert outputs for distillation
        expert_outputs = []
        for expert in self.moe.experts:
            expert_outputs.append(expert(x))
        
        # Get gating weights
        gates = self.moe.get_gate_values(x)
        
        # Calculate distillation loss
        distillation_loss = self.compute_distillation_loss(expert_outputs, gates)
        
        # Combine losses
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * distillation_loss
        
        # Backward pass and optimization
        total_loss.backward()
        self.moe.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'distillation_loss': distillation_loss.item() if isinstance(distillation_loss, torch.Tensor) else distillation_loss
        }
```

The key innovation in MoDE is the mutual distillation process. This is implemented in the `compute_distillation_loss` method, where each expert learns from all other experts through knowledge distillation. The method calculates the KL divergence between the softened outputs of each pair of experts, weighted by their gating values.

The distillation strength is controlled by the `alpha` parameter, which balances the standard classification loss and the distillation loss. The `distillation_temp` parameter adjusts the "softness" of the probability distributions used in distillation.

## Key Features

- Basic implementation of MoE (Mixture of Experts)
- Implementation of MoDE (Mixture of Distilled Experts)
- Support for experiments on computer vision datasets
- Model interpretation capabilities

## Installation

To install the required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

- `model/`: MoDE and MoE model implementations
- `utils/`: Utility functions
- `experiments/`: Experiment scripts
- `examples/`: Usage examples

## References

This implementation is based on the [MoDE: A Mixture-of-Experts Model with Mutual Distillation among the Experts](https://arxiv.org/abs/2402.00893) paper. 