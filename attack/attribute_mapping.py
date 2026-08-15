"""
Translate between PLY-style attribute names (``f_dc_0``, ``scale_1``, ...)
and the internal tensors of a MedGS ``GaussianModel`` (``_features_dc``,
``_scaling``, ...).

This indirection lets the CLI accept user-friendly names while the attack
code reads and writes the actual parameter tensors.
"""
import torch


def build_attribute_map(gaussians):
    """Build a dict mapping PLY field name -> ``(tensor_name, sub_index)``.

    Example entries::

        "x"        -> ("_xyz",          (0,))
        "f_dc_2"   -> ("_features_dc",  (0, 2))
        "scale_1"  -> ("_scaling",      (1,))

    Only attributes that the given ``GaussianModel`` actually exposes are
    included.
    """
    mapping = {}

    if hasattr(gaussians, "_xyz"):
        for i, name in enumerate(["x", "y", "z"]):
            mapping[name] = ("_xyz", (i,))

    if hasattr(gaussians, "_features_dc"):
        for i in range(3):
            mapping[f"f_dc_{i}"] = ("_features_dc", (0, i))

    if hasattr(gaussians, "_features_rest"):
        rest = gaussians._features_rest
        if rest.numel() > 0:
            K = rest.shape[1]
            for f_idx in range(3 * K):
                mapping[f"f_rest_{f_idx}"] = (
                    "_features_rest", (f_idx % K, f_idx // K)
                )

    if hasattr(gaussians, "_opacity"):
        mapping["opacity"] = ("_opacity", (0,))

    if hasattr(gaussians, "_scaling"):
        for i in range(gaussians._scaling.shape[1]):
            mapping[f"scale_{i}"] = ("_scaling", (i,))

    if hasattr(gaussians, "_rotation"):
        for i in range(gaussians._rotation.shape[1]):
            mapping[f"rot_{i}"] = ("_rotation", (i,))

    return mapping


def get_attribute_slice(gaussians, attr_name, sub_idx):
    """Read a per-Gaussian slice of an attribute tensor."""
    return getattr(gaussians, attr_name)[(slice(None),) + tuple(sub_idx)]


def set_attribute_slice_(gaussians, attr_name, sub_idx, new_values):
    """In-place write to a per-Gaussian slice of an attribute tensor."""
    t = getattr(gaussians, attr_name)
    with torch.no_grad():
        t[(slice(None),) + tuple(sub_idx)] = new_values


def get_gradient_slice(grad_tensor, sub_idx):
    """Read the gradient slice that mirrors an attribute slice."""
    return grad_tensor[(slice(None),) + tuple(sub_idx)]
