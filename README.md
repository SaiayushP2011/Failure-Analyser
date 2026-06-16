# Failure-Analyser
ML Training Analyzer

A PyTorch-based ML training analyzer that trains a simple neural network to learn the linear relationship between two tensors(for demonstration purposes, the relationship shown is a 'y = 2x' equation). The analyser tracks loss trends, counts the number of improvements and regressions(for the loss), detects loss-worsening patterns, and generates a concise, human-readable summary.

## Key Features

- Train a simple feedforward neural network using PyTorch.
- Track loss values across training epochs.
- Count training improvements and regressions.
- Detect repeated loss-worsening trends.
- Perform a simple generalization test on unseen input.
- Generate an automatic summary and recommendations based on training metrics.