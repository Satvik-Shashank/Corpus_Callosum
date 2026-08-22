import numpy as np
from src.morphology import measure

def test_measure_uses_physical_spacing() -> None:
    mask = np.array([[0,1,1],[0,1,1]], dtype=bool)
    result = measure(mask, (2.0, 0.5))
    assert result.area_mm2 == 4.0
    assert result.anterior_posterior_length_mm == 1.0
