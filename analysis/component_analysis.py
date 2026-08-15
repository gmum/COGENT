"""
Connected-components analysis on difference images, gated by lung masks.

For each diff image:
  1. binarize via Otsu (or a fixed threshold);
  2. apply morphological closing + small-object removal;
  3. label connected components;
  4. keep only components whose >70% of pixels lie inside the lung mask;
  5. save the binary mask, a side-by-side visualization, and a CSV row
     per kept component (centroid, area, bbox).
"""
import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import filters, measure, morphology
from skimage.transform import resize


# A component must overlap the lung mask by at least this fraction to be kept.
LUNG_OVERLAP_THRESHOLD = 0.7

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _binarize(diff_img, fixed_threshold):
    """Return ``(binary_mask, threshold_used)`` for a single difference image."""
    if fixed_threshold is None:
        threshold = max(filters.threshold_otsu(diff_img), 1)
    else:
        threshold = fixed_threshold
    return diff_img > threshold, threshold


def _clean_mask(binary_mask, closing_radius, min_area):
    """Apply morphological closing and remove tiny components."""
    if closing_radius > 0:
        binary_mask = morphology.binary_closing(
            binary_mask, footprint=morphology.disk(closing_radius)
        )
    return morphology.remove_small_objects(binary_mask, min_size=min_area)


def _resize_mask_to(mask, target_shape):
    """Resize a boolean mask to ``target_shape`` with nearest-neighbor."""
    if mask.shape == target_shape:
        return mask
    return resize(
        mask.astype(np.uint8), target_shape,
        order=0, preserve_range=True, anti_aliasing=False,
    ).astype(bool)


def _draw_visualization(out_path, diff_img, diff_mask, threshold_used,
                        lung_mask, kept, rejected, source_name):
    """Save a three-panel preview: diff | diff mask | overlay with centroids."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(diff_img, cmap="gray")
    axes[0].set_title(f"Diff: {source_name}")
    axes[0].axis("off")

    axes[1].imshow(diff_mask, cmap="gray")
    axes[1].set_title(f"Diff mask (threshold={threshold_used})")
    axes[1].axis("off")

    axes[2].imshow(diff_img, cmap="gray")
    axes[2].imshow(
        np.ma.masked_where(~lung_mask, lung_mask),
        cmap="cool", alpha=0.25,
    )
    axes[2].set_title(f"Kept: {len(kept)} | Rejected: {len(rejected)}")

    for region in kept:
        cy, cx = region.centroid
        minr, minc, maxr, maxc = region.bbox
        axes[2].add_patch(patches.Rectangle(
            (minc, minr), maxc - minc, maxr - minr,
            linewidth=1.2, edgecolor="lime", facecolor="none"))
        axes[2].plot(cx, cy, "r+", markersize=14, markeredgewidth=2)

    for region in rejected:
        cy, cx = region.centroid
        minr, minc, maxr, maxc = region.bbox
        axes[2].add_patch(patches.Rectangle(
            (minc, minr), maxc - minc, maxr - minr,
            linewidth=1.0, edgecolor="gray",
            facecolor="none", linestyle="--"))
        axes[2].plot(cx, cy, "x", color="gray",
                     markersize=10, markeredgewidth=1.5)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def analyze_connected_components(diff_folder, output_folder, lung_masks_folder,
                                 threshold=None, min_area=20, closing_radius=3):
    """Find connected components in diff images, filtered by lung masks.

    Args:
        diff_folder:        folder with difference images.
        output_folder:      where to write masks, visualizations and
                            ``centroids.csv``.
        lung_masks_folder:  folder with lung masks (white = lungs). Must
                            contain the same number of files as ``diff_folder``;
                            matching is done by stem, falling back to sorted
                            index.
        threshold:          binarization threshold (0-255); ``None`` = Otsu
                            per image.
        min_area:           minimum component area in pixels.
        closing_radius:     radius of the morphological closing disk
                            (``0`` disables closing).

    Returns:
        ``pandas.DataFrame`` with one row per kept centroid.
    """
    Path(f"{output_folder}/masks").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/visualizations").mkdir(parents=True, exist_ok=True)

    diff_files = sorted([f for f in os.listdir(diff_folder)
                         if Path(f).suffix.lower() in _IMAGE_EXTENSIONS])
    mask_files = sorted([f for f in os.listdir(lung_masks_folder)
                         if Path(f).suffix.lower() in _IMAGE_EXTENSIONS])

    if len(diff_files) != len(mask_files):
        raise ValueError(
            f"Number of diff images ({len(diff_files)}) does not match "
            f"number of lung masks ({len(mask_files)}). "
            f"Check folders: {diff_folder} and {lung_masks_folder}"
        )

    mask_by_stem = {Path(f).stem: f for f in mask_files}

    centroid_rows = []
    rejected_total = 0

    for idx, diff_name in enumerate(diff_files):
        diff_path = os.path.join(diff_folder, diff_name)
        stem = Path(diff_name).stem

        diff_img = np.array(Image.open(diff_path).convert("L"))

        # Match by stem first, fall back to index if stems do not line up.
        mask_name = mask_by_stem.get(stem, mask_files[idx])
        mask_path = os.path.join(lung_masks_folder, mask_name)
        lung_mask = np.array(Image.open(mask_path).convert("L")) > 127
        lung_mask = _resize_mask_to(lung_mask, diff_img.shape)

        diff_mask, threshold_used = _binarize(diff_img, threshold)
        diff_mask = _clean_mask(diff_mask, closing_radius, min_area)

        labels = measure.label(diff_mask, connectivity=2)
        regions = measure.regionprops(labels)

        kept, rejected = [], []
        for region in regions:
            overlap = lung_mask[labels == region.label].mean()
            if overlap <= LUNG_OVERLAP_THRESHOLD:
                rejected.append(region)
                rejected_total += 1
                continue

            kept.append(region)
            cy, cx = region.centroid
            minr, minc, maxr, maxc = region.bbox
            centroid_rows.append({
                "file": diff_name,
                "component_id": int(region.label),
                "centroid_x": round(float(cx), 2),
                "centroid_y": round(float(cy), 2),
                "area_pix": int(region.area),
                "bbox_x": int(minc),
                "bbox_y": int(minr),
                "bbox_w": int(maxc - minc),
                "bbox_h": int(maxr - minr),
            })

        Image.fromarray((diff_mask * 255).astype(np.uint8)).save(
            f"{output_folder}/masks/{stem}_mask.png")

        _draw_visualization(
            out_path=f"{output_folder}/visualizations/{stem}_viz.png",
            diff_img=diff_img, diff_mask=diff_mask,
            threshold_used=threshold_used,
            lung_mask=lung_mask, kept=kept, rejected=rejected,
            source_name=diff_name,
        )

        print(f"  {diff_name:30s} | thr={threshold_used:3d} | "
              f"kept={len(kept):3d} | rejected={len(rejected):3d}")

    df = pd.DataFrame(centroid_rows)
    csv_path = f"{output_folder}/centroids.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nWrote {len(df)} centroids to {csv_path}")
    print(f"Total rejected (outside lungs): {rejected_total}")
    print(f"Masks:           {output_folder}/masks/")
    print(f"Visualizations:  {output_folder}/visualizations/")

    return df


if __name__ == "__main__":
    df = analyze_connected_components(
        diff_folder="output/diffs",
        output_folder="analysis",
        lung_masks_folder=(
            "1.3.6.1.4.1.14519.5.2.1.6279.6001."
            "100398138793540579077826395208/lungs_masks_png"
        ),
        threshold=None,
        min_area=20,
        closing_radius=5,
    )
    print("\nPreview:")
    print(df.head(10))
