
import json
import nibabel as nib
import numpy as np
import torch

from pathlib import Path
from .base import BaseDataset

from utils.sybil_utils import apply_sybil_transforms, get_spacing

import logging
log = logging.getLogger(__name__)


class TestCTDataset(BaseDataset):
    def __init__(
            self,
            name: str,
            split: str,
            path_data: str
        ):
        super().__init__()
        self.name = name
        self.split = split # ignored for this dataset
        self.path_data = Path(path_data)

        files = list((self.path_data).glob("*.nii*"))
        self.data = [{'image': str(f)} for f in files]
        self.length = len(self.data)

    def preprocess_scan(self, scan: nib.Nifti1Image) -> torch.Tensor:
        volume = scan.get_fdata()
        affine = scan.affine
        spacing = get_spacing(affine)

        volume = torch.from_numpy(volume).float()
        volume = volume.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        volume, resampled_shape = apply_sybil_transforms(volume, input_spacing=spacing)
        return volume, spacing, resampled_shape

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index = index
        scan = nib.load(self.data[index]['image'])
        volume, original_spacing, resampled_shape = self.preprocess_scan(scan)
        volume.requires_grad = False

        return volume[0], index, original_spacing, resampled_shape
