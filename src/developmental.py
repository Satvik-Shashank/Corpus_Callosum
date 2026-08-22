"""Age-aware research analysis with no diagnostic interpretation."""
from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class ReferenceComparison:
    z_score: float
    message: str

def compare_to_reference(value: float, age_months: float, reference_values: Sequence[float], reference_ages_months: Sequence[float]) -> ReferenceComparison:
    """Compare with an explicitly supplied, age-matched research reference set."""
    ages, values = np.asarray(reference_ages_months), np.asarray(reference_values)
    nearby = values[np.abs(ages-age_months) <= 1.0]
    if nearby.size < 5: raise ValueError("Insufficient age-matched reference observations.")
    z = (value-nearby.mean()) / (nearby.std(ddof=1)+1e-8)
    return ReferenceComparison(float(z), "Research comparison only; not diagnostic.")
