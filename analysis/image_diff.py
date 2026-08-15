"""Generate per-pixel absolute-difference images between two folders.

For every image present in both folders (matched by file name), writes
``|A - B|`` to the output folder as ``diff_<name>``.
"""
import os

import cv2


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def generate_image_diffs(folder_a, folder_b, output_folder):
    """Write ``|A - B|`` for each image present under both folders.

    For every image file in ``folder_a``, if a file with the same name
    exists in ``folder_b`` and both have the same shape, the absolute
    difference is written to ``output_folder/diff_<name>``.

    Args:
        folder_a:      first input folder (e.g. attacked renders).
        folder_b:      second input folder (e.g. clean renders).
        output_folder: destination folder; created if missing.
    """
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(folder_a)
             if f.lower().endswith(_IMAGE_EXTENSIONS)]

    for filename in files:
        path_a = os.path.join(folder_a, filename)
        path_b = os.path.join(folder_b, filename)

        if not os.path.exists(path_b):
            print(f"Skipping: {filename} is missing from {folder_b}")
            continue

        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        if img_a is None or img_b is None:
            print(f"Skipping: failed to read {filename}")
            continue

        if img_a.shape != img_b.shape:
            print(f"Skipping: {filename} has different shapes in the two folders.")
            continue

        diff = cv2.absdiff(img_a, img_b)

        out_path = os.path.join(output_folder, f"diff_{filename}")
        cv2.imwrite(out_path, diff)
        print(f"Wrote: {filename}")


if __name__ == "__main__":
    generate_image_diffs(
        folder_a="output/1_attacked/render",
        folder_b="output/1/render",
        output_folder="output/diffs",
    )
