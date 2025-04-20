import torch
import torch.nn as nn
import numpy as np


def create_target_distribution(y_true, num_classes, epsilon=0.05):
    """
    Create a target distribution close to uniform but slightly favor the correct class.
    
    Each sample's distribution:
      - correct class: 1/C + epsilon
      - other classes: 1/C - epsilon/(C-1)

    This ensures the distribution sums to 1 and does not require renormalization.

    Args:
        y_true (Tensor): True labels, shape (batch_size,)
        num_classes (int): Number of classes
        epsilon (float): Adjustment factor to favor the correct class

    Returns:
        Tensor: Target distributions, shape (batch_size, num_classes)
    """

    # Start with a uniform distribution
    batch_size = y_true.size(0)
    target = torch.full((batch_size, num_classes), 1.0 / num_classes, device=y_true.device)

    # Increase the probability of the correct class by epsilon
    target[range(batch_size), y_true] += epsilon

    # Subtract epsilon/(C-1) from all other classes
    # Create a mask for non-correct classes
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[range(batch_size), y_true] = False
    target[mask] -= epsilon / (num_classes - 1)

    return target


class KLDivLossWithTarget(nn.Module):
    def __init__(self, num_classes, epsilon=0.05):
        super(KLDivLossWithTarget, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, log_probs, y_true):
        """
        Compute KL divergence between model predictions and target distribution.

        Args:
            logits (Tensor): Model output logits, shape (batch_size, num_classes)
            y_true (Tensor): True labels, shape (batch_size,)

        Returns:
            Tensor: KL divergence loss
        """
        # Compute log probabilities from logits
        # log_probs = self.log_softmax(logits)
        
        # Create target distributions
        target_dist = create_target_distribution(y_true, self.num_classes, self.epsilon)
        
        # Compute KL divergence
        loss = self.kl_div(log_probs, target_dist)
        
        return loss