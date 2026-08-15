from torch.utils.data import Dataset
from abc import ABC

class BaseDataset(Dataset, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)