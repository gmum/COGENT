from torch import nn
import torch
import json

from .base import ClassifierBase
from utils.iterp import torch_interp
from pathlib import Path

import sys
sys.path.append("src_sybil")
from sybil.models.sybil import SybilNet


SYBIL_YEAR1_60P = None # Placeholder to compute


class SimpleClassifierGroupTorch(nn.Module):
    """
    A class to represent a calibrator for prediction models.
    Behavior and coefficients are taken from the sklearn.calibration.CalibratedClassifierCV class.
    Make a custom class to avoid sklearn versioning issues.

    Adapted from Sybil to work with PyTorch.
    """

    def __init__(self, calibrators):
        super().__init__()
        self.calibrators = calibrators

    def predict_proba(self, X, expand=False):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_probabilities,)
            The input probabilities to recalibrate.
        expand : bool, default=False
            Whether to return the probabilities for each class separately.
            This is intended for binary classification which can be done in 1D,
            expand=True will return a 2D array with shape (n_probabilities, 2).

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. Classes are ordered by
            lexicographic order.
        """
        proba = torch.stack([calibrator.transform(X) for calibrator in self.calibrators])
        pos_prob = torch.mean(proba, dim=0)

        if expand and len(self.calibrators) == 1:
            return torch.tensor([1.-pos_prob, pos_prob])
        else:
            return pos_prob

    @classmethod
    def from_json(cls, json_list):
        return cls([SimpleIsotonicRegressorTorch.from_json(json_dict) for json_dict in json_list])

    @classmethod
    def from_json_grouped(cls, json_path):
        """
        We store calibrators in a diction of {year (str): [calibrators]}.
        This is a convenience method to load that dictionary from a file path.
        """
        json_dict = json.load(open(json_path, "r"))
        output_dict = {key: cls.from_json(json_list) for key, json_list in json_dict.items()}
        return output_dict


class SimpleIsotonicRegressorTorch(nn.Module):
    def __init__(self, coef, intercept, x0, y0, x_min=-torch.inf, x_max=torch.inf):
        """
        Adapted from Sybil to work with PyTorch.
        """
        super().__init__()
        self.coef = coef
        self.intercept = intercept
        self.x0 = x0
        self.y0 = y0
        self.x_min = x_min
        self.x_max = x_max

    def transform(self, X):
        T = X @ self.coef + self.intercept
        T = torch.clip(T, self.x_min, self.x_max)
        return torch_interp(T, self.x0, self.y0)

    @classmethod
    def from_json(cls, json_dict):
        return cls(
            torch.tensor(json_dict["coef"]).cuda(),
            torch.tensor(json_dict["intercept"]).cuda(),
            torch.tensor(json_dict["x0"]).cuda(),
            torch.tensor(json_dict["y0"]).cuda(),
            json_dict["x_min"],
            json_dict["x_max"]
        )

    def __repr__(self):
        return f"SimpleIsotonicRegressorTorch(x={self.x0}, y={self.y0})"



class Sybil(ClassifierBase):

    def __init__(
            self,
            path_ckpt: str,
            calibrator_path: str,
            x_min_hu: int = -1025,
            x_max_hu: int = 3071,
            num_years: int = 6,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        path_ckpt = Path(path_ckpt)
        calibrator_path = Path(calibrator_path)
        self.x_min_hu = x_min_hu
        self.x_max_hu = x_max_hu
        self.num_years = num_years

        self.x_orig = None
        self.mask_locations = None

        assert path_ckpt.exists(), f"ckpt_dir {path_ckpt} does not exist"
        assert calibrator_path.exists(), f"calibrator_path {calibrator_path} does not exist"

        self.model = self.load_model(path_ckpt)
        self.calibrator = SimpleClassifierGroupTorch.from_json_grouped(calibrator_path)

        self.eval()


    def load_model(self, path: Path):
        ckpt_model = torch.load(path, map_location="cpu")
        args = ckpt_model.prob_of_failure_layer.args # Extract args from checkpoint
        model = SybilNet(args)
        model.load_state_dict(ckpt_model.state_dict())  # type: ignore
        return model


    def calibrate(self, scores):
        calibrated_scores = []
        for YEAR in range(scores.shape[1]):
            probs = scores[:, YEAR].reshape(-1, 1)
            probs = self.calibrator["Year{}".format(YEAR + 1)].predict_proba(probs)[:, -1]
            calibrated_scores.append(probs)

        return torch.stack(calibrated_scores, axis=1)


    def forward_all_years(self, x):
        """Get the sigmoid of the base hazard and calibrated risk for all years"""
        y = self.model(x)
        scores = self.calibrate(y["logit"].sigmoid())
        base_hazard = y["base_hazard"].sigmoid()

        y = torch.concat([base_hazard, scores], dim=1)
        y = y.clip(0, 1) # Remove this to make the model differentiable

        return y


    def forward(self, x):
        """Get the year 1 risk as the forward output"""
        scores = self.forward_all_years(x)
        return scores[:, 1]


    def pred_prob(self, x):
        y = self.forward(x)
        # This is a hack to make binary classification compatible with the rest of the code
        return torch.stack([y, 1-y], dim=1).float()
    

    def pred_label(self, x):
        x = self.pred_prob(x)
        # 0 is low risk, 1 is high risk
        return (x[:, 0] > SYBIL_YEAR1_60P).long()


    def load_ckpt(self, *args, **kwargs):
        pass
        