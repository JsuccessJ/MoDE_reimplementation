# MoDE (Mixture of Distilled Experts) Implementation

This project is an implementation of the MoDE (Mixture of Distilled Experts) model. MoDE is an improved version of MoE (Mixture of Experts) that uses expert models, enhancing the model's generalization ability through mutual knowledge distillation.

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

## Usage

```python
from mode import MoDE, MoE

# Initialize model
mode_model = MoDE(num_experts=2, input_dim=784, hidden_dim=256, output_dim=10)

# Training
mode_model.train(train_loader, epochs=10)

# Evaluation
accuracy = mode_model.evaluate(test_loader)
```

## Project Structure

- `model/`: MoDE and MoE model implementations
- `utils/`: Utility functions
- `experiments/`: Experiment scripts
- `examples/`: Usage examples

## References

This implementation is based on the MoDE paper. 