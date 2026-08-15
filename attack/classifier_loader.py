"""Load the Sybil ensemble classifier with weights frozen for attack use."""
import gc
import os

import torch
import torch.nn as nn


def load_sybil_classifier(config_dir, config_name, device, n_models_to_load=1):
    """Instantiate a Sybil ensemble via Hydra and freeze its parameters.

    Freezing only prevents weight updates - gradient still flows through
    the classifier, so the attack can propagate gradients back into the
    Gaussian parameters. Calibration and clipping from the public
    ``forward_all_years`` are intentionally not used; we attack the raw
    logits instead.

    Args:
        config_dir:        absolute path to the Hydra config directory.
        config_name:       Hydra config name (without ``.yaml``).
        device:            torch device for the classifier weights.
        n_models_to_load:  how many ensemble members to keep in memory.
                           ``SybilEnsemble.__init__`` loads all 5
                           checkpoints (~660 MB). For adversarial-attack
                           purposes a single member is usually enough.
                           ``0`` means "keep all".

    Returns:
        The classifier in eval mode with all parameters frozen.
    """
    try:
        from hydra import initialize_config_dir, compose
        from hydra.utils import instantiate as hydra_instantiate
    except ImportError:
        raise ImportError(
            "hydra-core is required; install with "
            "`pip install hydra-core omegaconf`"
        )

    config_dir_abs = os.path.abspath(config_dir)
    if not os.path.isdir(config_dir_abs):
        raise FileNotFoundError(f"Missing config directory: {config_dir_abs}")

    print(f"  config_dir:  {config_dir_abs}")
    print(f"  config_name: {config_name}")

    with initialize_config_dir(version_base=None, config_dir=config_dir_abs):
        config = compose(config_name=config_name)
    classifier = hydra_instantiate(config.classifier)

    # Drop unused ensemble members BEFORE moving to the target device.
    if (hasattr(classifier, "ensemble") and n_models_to_load > 0
            and n_models_to_load < len(classifier.ensemble)):
        total = len(classifier.ensemble)
        kept = list(classifier.ensemble)[:n_models_to_load]
        classifier.ensemble = nn.ModuleList(kept)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  Ensemble: kept {n_models_to_load}/{total} models "
              f"(remaining freed from RAM)")

    classifier = classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    # ``SybilEnsemble.calibrator.from_json`` hardcodes ``.cuda()`` - move
    # the calibration tensors explicitly so the whole pipeline lives on
    # the requested device.
    if hasattr(classifier, "calibrator") and isinstance(classifier.calibrator, dict):
        for _, cal_group in classifier.calibrator.items():
            cal_group.to(device)

    n_params = sum(p.numel() for p in classifier.parameters())
    n_models = len(classifier.ensemble) if hasattr(classifier, "ensemble") else 1
    print(f"  Ensemble members:  {n_models}")
    print(f"  Parameters:        {n_params:,} (frozen, device: {device})")
    return classifier
