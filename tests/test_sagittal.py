import numpy as np
from src.sagittal import SagittalConfig, select_mid_sagittal

def test_selects_symmetric_centre_plane() -> None:
    volume=np.ones((11,8,8),dtype=np.float32) + np.arange(8, dtype=np.float32)[None,None,:]
    volume[5] = np.tile(np.array([1,2,3,4,4,3,2,1]),(8,1))
    _,selection=select_mid_sagittal(volume,config=SagittalConfig(search_radius=5,minimum_mask_pixels=1))
    assert selection.index == 5
