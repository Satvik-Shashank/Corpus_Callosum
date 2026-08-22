"""Losses suitable for imbalanced binary CC segmentation."""
import torch
import torch.nn.functional as F

def dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for two-class logits and integer targets."""
    probabilities = F.softmax(logits, dim=1)[:, 1]
    target = target.float(); intersection = (probabilities*target).sum((1,2))
    return 1 - ((2*intersection+epsilon)/(probabilities.sum((1,2))+target.sum((1,2))+epsilon)).mean()
