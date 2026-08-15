from torch import nn
import torch
import torch.nn.functional as F
from pathlib import Path

from .base import ClassifierBase


import sys
sys.path.append("src_sybil")
from sybil.models.sybil import SybilNet

from .sybil import SimpleClassifierGroupTorch
from utils.sybil_utils import _croppad


import logging
logger = logging.getLogger(__name__)


SYBIL_ENSEMBLE_YEAR1_60P = 0.0071


class SybilEnsemble(ClassifierBase):

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

        ckpt_dir = Path(path_ckpt)
        calibrator_path = Path(calibrator_path)
        self.x_min_hu = x_min_hu
        self.x_max_hu = x_max_hu
        self.num_years = num_years

        self.x_orig = None
        self.mask_locations = None

        assert ckpt_dir.exists(), f"ckpt_dir {ckpt_dir} does not exist"
        assert calibrator_path.exists(), f"calibrator_path {calibrator_path} does not exist"

        # Load ensemble model
        self.ensemble = torch.nn.ModuleList()
        ckpt_paths = list(ckpt_dir.glob("sybil_*.pt"))
        logger.info("Found %d checkpoints in %s", len(ckpt_paths), ckpt_dir)
        ckpt_paths.sort()
        for path in ckpt_paths:
            self.ensemble.append(self.load_model(path))

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
        scores = []
        base_hazard = []
        for model in self.ensemble:
            scores.append(model(x)["logit"])
            base_hazard.append(model(x)["base_hazard"])
        scores = torch.mean(torch.stack(scores, dim=0), dim=0).sigmoid()
        scores = self.calibrate(scores)
        base_hazard = torch.mean(torch.stack(base_hazard, dim=0), dim=0).sigmoid()

        y = torch.concat([base_hazard, scores], dim=1)
        y = y.clip(0, 1) # Remove this to make the model differentiable

        return y


    def forward(self, x):
        """Get the year 1 risk as the forward output"""
        scores = self.forward_all_years(x)
        return scores[:, 1]


    @torch.no_grad()
    def get_attention(self, x: torch.Tensor, resampled_shape):
        batch_size = x.shape[0]
        attentions = []
        for i in range(len(self.ensemble)):
            out = self.ensemble[i](x)
            a1 = out["image_attention_1"]
            v1 = out["volume_attention_1"]

            a1 = torch.exp(a1).mean(0)
            v1 = torch.exp(v1).mean(0)

            attention = a1 * v1.unsqueeze(-1)
            attention = attention.view(batch_size, 1, *out['activ'].shape[-3:]).permute(0, 1, 3, 4, 2)

            # Upsample to model input
            attention = F.interpolate(attention, size=x.shape[-3:], mode="trilinear", align_corners=True)
            # Undo the crop/pad
            attention = _croppad(attention, target_shape=resampled_shape[-3:])

            attentions.append(attention)

        attention = torch.stack(attentions, dim=0).mean(dim=0)
        return attention


    def pred_prob(self, x):
        y = self.forward(x)
        # This is a hack to make binary classification compatible with the rest of the code
        return torch.stack([y, 1-y], dim=1).float()
    

    def pred_label(self, x):
        x = self.pred_prob(x)
        # 0 is low risk, 1 is high risk
        return (x[:, 0] > SYBIL_ENSEMBLE_YEAR1_60P).long()

    def load_ckpt(self, *args, **kwargs):
        pass
