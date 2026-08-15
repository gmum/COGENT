from abc import ABC, abstractmethod
from torch import nn

class ClassifierBase(ABC, nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval()

    @abstractmethod
    def load_ckpt(self, *args, **kwargs):
        pass

    def pred_prob(self, *args, **kwargs):
        pass
    
    def pred_label(self, *args, **kwargs):
        pass