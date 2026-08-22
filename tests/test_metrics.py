import numpy as np
from evaluation.metrics import dice_coefficient

def test_dice_for_identical_masks() -> None:
    mask = np.array([[0,1],[1,0]])
    assert dice_coefficient(mask, mask) == 1.0
