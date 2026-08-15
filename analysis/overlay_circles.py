"""Overlay circles on original images at centroids extracted by component analysis."""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _extract_zero_based_index(diff_filename):
    """Parse a diff filename like ``'diff_0001.png'`` -> ``0`` (zero-based)."""
    m = re.search(r"(\d+)", Path(diff_filename).stem)
    if m is None:
        raise ValueError(f"No numeric index found in: {diff_filename}")
    return int(m.group(1)) - 1


def _compute_radius(row, radius_mode, scale, min_radius):
    """Compute the circle radius for one centroid row according to ``radius_mode``."""
    if radius_mode == "area":
        r = np.sqrt(row["area_pix"] / np.pi) * scale
    elif radius_mode == "bbox":
        r = max(row["bbox_w"], row["bbox_h"]) / 2 * scale
    elif radius_mode == "constant":
        r = float(scale)
    else:
        raise ValueError(f"Unknown radius_mode: {radius_mode}")
    return max(r, min_radius)


def draw_circles_on_originals(csv_path, originals_folder, output_folder,
                              radius_mode="area", scale=1.0,
                              min_radius=3, thickness=2,
                              color=(255, 0, 0)):
    """Draw circles around centroids on the matching original images.

    Mapping: each diff filename (e.g. ``"diff_0001.png"``) is mapped to
    the 1st (zero-based: 0) image of ``originals_folder`` after sorting
    by name.

    All originals are written to ``output_folder``, even those without
    any centroids - in that case the file is just a copy of the input.

    Args:
        csv_path:         path to ``centroids.csv`` from
                          ``analyze_connected_components``.
        originals_folder: folder with the original images to annotate.
        output_folder:    where to write the annotated copies.
        radius_mode:      ``"area"`` (radius from component area),
                          ``"bbox"`` (from bounding-box dimensions), or
                          ``"constant"`` (uses ``scale``).
        scale:            multiplier on the computed radius, or absolute
                          radius in ``"constant"`` mode.
        min_radius:       minimum drawn radius in pixels.
        thickness:        circle outline thickness in pixels.
        color:            RGB tuple for the circle outline.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    originals = sorted([f for f in os.listdir(originals_folder)
                        if Path(f).suffix.lower() in _IMAGE_EXTENSIONS])

    # Map "original image index" -> the centroid rows that belong to it.
    rows_by_index = {}
    for diff_name, group in df.groupby("file"):
        try:
            idx = _extract_zero_based_index(diff_name)
        except ValueError as e:
            print(f"  Skipping {diff_name}: {e}")
            continue
        if 0 <= idx < len(originals):
            rows_by_index[idx] = group
        else:
            print(f"  Skipping {diff_name}: no original at index {idx}")

    n_written = 0
    n_circles = 0
    n_blank = 0

    for idx, original_name in enumerate(originals):
        src_path = os.path.join(originals_folder, original_name)
        img = Image.open(src_path).convert("RGB")
        rows = rows_by_index.get(idx)

        n_drawn = 0
        if rows is not None and len(rows) > 0:
            draw = ImageDraw.Draw(img)
            for _, row in rows.iterrows():
                cx = float(row["centroid_x"])
                cy = float(row["centroid_y"])
                r = _compute_radius(row, radius_mode, scale, min_radius)
                draw.ellipse(
                    [(cx - r, cy - r), (cx + r, cy + r)],
                    outline=color, width=thickness,
                )
                n_drawn += 1

        stem = Path(original_name).stem
        ext = Path(original_name).suffix
        out_path = os.path.join(output_folder, f"{stem}_circles{ext}")
        img.save(out_path)

        n_written += 1
        n_circles += n_drawn
        if n_drawn > 0:
            print(f"  {original_name}: {n_drawn} circles -> {out_path}")
        else:
            n_blank += 1
            print(f"  {original_name}: no centroids (saved original copy)")

    print(f"\nDone.")
    print(f"  Images written:           {n_written}")
    print(f"  With circles:             {n_written - n_blank}")
    print(f"  Without centroids:        {n_blank}")
    print(f"  Total circles drawn:      {n_circles}")
    print(f"  Output folder:            {output_folder}")


if __name__ == "__main__":
    draw_circles_on_originals(
        csv_path="analysis/centroids.csv",
        originals_folder="output/1_attacked/render",
        output_folder="annotated",
        radius_mode="area",
        scale=1.5,
        min_radius=10,
        thickness=2,
        color=(255, 0, 0),
    )
