import numpy as np
from src.annotation import CandidateConfig, generate_candidate_mask

def test_candidate_is_binary_and_bounded() -> None:
    image=np.zeros((20,20)); image[8:12,8:12]=10
    mask=generate_candidate_mask(image,CandidateConfig(polarity="bright"))
    assert mask.dtype == bool and mask.shape == image.shape and mask.any()
