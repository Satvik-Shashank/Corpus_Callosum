"""Segmentation metrics."""
import numpy as np

def dice_coefficient(prediction: np.ndarray, target: np.ndarray, epsilon: float = 1e-8) -> float:
    """Calculate binary Dice, rejecting incompatible arrays."""
    if prediction.shape != target.shape: raise ValueError("Prediction and target shapes differ.")
    prediction, target = prediction.astype(bool), target.astype(bool)
    return float((2*np.logical_and(prediction,target).sum()+epsilon)/(prediction.sum()+target.sum()+epsilon))
