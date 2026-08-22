"""Label-source inspection. Numeric label values are never interpreted without a LUT."""
from pathlib import Path
from typing import Optional
import nibabel as nib
import numpy as np

def inspect_label_source(path: str|Path, label_lookup: Optional[dict[int,str]] = None) -> dict[str,object]:
    """Report values and whether semantics explicitly verify a corpus-callosum class."""
    image=nib.load(str(path)); data=np.asanyarray(image.dataobj); values=sorted(int(v) for v in np.unique(data))
    labels=label_lookup or {}; cc_values=[key for key,value in labels.items() if "corpus callosum" in value.lower()]
    return {"path":str(path),"shape":list(data.shape),"values":values,"label_lookup_supplied":bool(label_lookup),"verified_cc_label_values":cc_values,"cc_semantics_verified":bool(cc_values),"warning":"Do not use this image as CC ground truth unless the lookup is authoritative and verified."}
