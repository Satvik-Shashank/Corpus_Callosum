"""Dataset validation for paired MRI planes and expert CC masks."""
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset

class CCSagittalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory dataset requiring binary masks that share image dimensions."""
    def __init__(self, images: Sequence[Path], masks: Sequence[Path]) -> None:
        if len(images) != len(masks): raise ValueError("Images and masks must be paired.")
        self.images, self.masks = list(images), list(masks)
    def __len__(self) -> int: return len(self.images)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = np.load(self.images[index]), np.load(self.masks[index])
        if image.shape != mask.shape or not np.isin(mask,[0,1]).all(): raise ValueError("Expected aligned binary expert mask.")
        return torch.from_numpy(image.astype(np.float32))[None], torch.from_numpy(mask.astype(np.int64))
