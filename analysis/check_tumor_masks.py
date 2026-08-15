"""Simple utility to list mask slices that contain tumor pixels.

A slice is considered positive if it contains at least ``min_white_pixels``
pixels with intensity >= ``threshold``.
"""
import argparse
import csv
import os
from pathlib import Path

from PIL import Image


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_positive_mask_slices(masks_dir, threshold=128, min_white_pixels=1):
    """Return per-file stats and positive slices for a mask directory.

    Args:
        masks_dir: path to folder containing per-slice mask images.
        threshold: pixel value threshold for white tumor pixels.
        min_white_pixels: minimum number of white pixels to mark a slice positive.

    Returns:
        results: list of dicts with keys:
            filename, white_pixels, has_tumor
    """
    masks_path = Path(masks_dir)
    if not masks_path.is_dir():
        raise FileNotFoundError(f"Masks directory does not exist: {masks_dir}")

    files = sorted(
        p for p in masks_path.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    results = []
    for p in files:
        img = Image.open(p).convert("L")
        hist = img.histogram()
        white_pixels = int(sum(hist[threshold:]))
        has_tumor = white_pixels >= int(min_white_pixels)
        results.append(
            {
                "filename": p.name,
                "white_pixels": white_pixels,
                "has_tumor": has_tumor,
            }
        )

    return results


def _write_csv(results, output_csv):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "white_pixels", "has_tumor"],
        )
        writer.writeheader()
        writer.writerows(results)


def _print_summary(results, show_negative=False):
    n_total = len(results)
    positives = [r for r in results if r["has_tumor"]]
    negatives = [r for r in results if not r["has_tumor"]]

    print(f"Total mask slices: {n_total}")
    print(f"Tumor-positive:    {len(positives)}")
    print(f"Tumor-negative:    {len(negatives)}")

    print("\nPositive slices:")
    if positives:
        for r in positives:
            print(f"  {r['filename']}  (white_pixels={r['white_pixels']})")
    else:
        print("  none")

    if show_negative:
        print("\nNegative slices:")
        if negatives:
            for r in negatives:
                print(f"  {r['filename']}  (white_pixels={r['white_pixels']})")
        else:
            print("  none")


def parse_args():
    parser = argparse.ArgumentParser(
        description="List mask slices containing tumor pixels.",
    )
    parser.add_argument(
        "--masks_dir",
        required=True,
        help="Directory with per-slice mask images.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Pixel threshold for counting white/tumor pixels (default: 128).",
    )
    parser.add_argument(
        "--min_white_pixels",
        type=int,
        default=1,
        help="Minimum white pixels to mark a slice as tumor-positive (default: 1).",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional CSV output path for per-slice counts.",
    )
    parser.add_argument(
        "--show_negative",
        action="store_true",
        help="Also print slices that do not contain tumor pixels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results = find_positive_mask_slices(
        masks_dir=args.masks_dir,
        threshold=args.threshold,
        min_white_pixels=args.min_white_pixels,
    )
    _print_summary(results, show_negative=args.show_negative)

    if args.output_csv:
        _write_csv(results, args.output_csv)
        print(f"\nSaved CSV: {os.path.abspath(args.output_csv)}")


if __name__ == "__main__":
    main()
