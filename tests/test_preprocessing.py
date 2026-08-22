import numpy as np
import pytest
from src.preprocessing import candidate_brain_mask, normalize_intensity, validate_volume

def test_normalization_preserves_shape_and_background() -> None:
    data=np.zeros((5,5,5),dtype=np.float32); data[1:4,1:4,1:4]=np.arange(27).reshape(3,3,3)+1
    mask=candidate_brain_mask(data,10); result=normalize_intensity(data,mask)
    assert result.shape == data.shape and np.all(result[~mask] == 0)
def test_invalid_volume_fails() -> None:
    with pytest.raises(ValueError): validate_volume(np.zeros((3,3,3)))
