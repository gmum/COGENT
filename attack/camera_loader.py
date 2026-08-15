"""Match cameras from ``cameras.json`` with mask images on disk."""
import json
import os

from .geometry import build_world_to_camera


def _find_mask_for_image(img_name, masks_dir, mask_files):
    """Locate a mask file that corresponds to ``img_name``.

    Tries the following strategies in order:
      1. Exact match with common image extensions.
      2. Zero-padded numeric variants of the stem (and ``stem - 1``).
      3. Prefix match for files like ``000123_<anything>.png``.
    """
    mask_files_set = set(mask_files)
    s = str(img_name)
    s_stem = os.path.splitext(s)[0]

    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        for candidate in (s + ext, s_stem + ext):
            if candidate in mask_files_set:
                return os.path.join(masks_dir, candidate)

    try:
        n = int(s_stem)
    except (TypeError, ValueError):
        return None

    for n_try in (n, n - 1):
        if n_try < 0:
            continue
        for pad in (1, 2, 3, 4, 5, 6):
            for ext in (".png", ".jpg", ".jpeg"):
                name = f"{n_try:0{pad}d}{ext}"
                if name in mask_files_set:
                    return os.path.join(masks_dir, name)
        for pad in (3, 4, 5, 6):
            prefix = f"{n_try:0{pad}d}_"
            cands = [f for f in mask_files if f.startswith(prefix)]
            if cands:
                return os.path.join(masks_dir, sorted(cands)[0])
    return None


def load_cameras_with_masks(cameras_json_path, masks_dir):
    """Return camera dicts that have a matching mask file on disk.

    Each returned dict contains: ``img_name``, ``w2c``, ``fx``, ``fy``,
    ``width``, ``height``, ``mask_path``.
    """
    if not os.path.isdir(masks_dir):
        print(f"ERROR: masks directory does not exist: {masks_dir}")
        return []

    with open(cameras_json_path) as f:
        raw = json.load(f)

    mask_files = [
        f for f in os.listdir(masks_dir)
        if os.path.isfile(os.path.join(masks_dir, f))
        and f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"  Files in masks_dir: {len(mask_files)}")

    cameras = []
    skipped = 0
    for entry in raw:
        mask_path = _find_mask_for_image(entry["img_name"], masks_dir, mask_files)
        if mask_path is None:
            skipped += 1
            continue
        cameras.append(dict(
            img_name=str(entry["img_name"]),
            w2c=build_world_to_camera(entry["position"], entry["rotation"]),
            fx=float(entry["fx"]),
            fy=float(entry["fy"]),
            width=int(entry["width"]),
            height=int(entry["height"]),
            mask_path=mask_path,
        ))

    print(f"  Cameras with mask: {len(cameras)}, without mask: {skipped}")
    return cameras
