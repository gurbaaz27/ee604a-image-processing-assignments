"""
Usage:
    pip install opencv-python numpy pillow
    python thief.py path-to-image
"""
__author__ = "Gurbaaz Singh Nandra"
__rollno__ = 190349
__filename__ = "thief.py"
__description__ = """ python program
file thief.py, which accepts command line file path for input image and directly stores the
enhanced-cctv3.jpg file when the command {$python thief.py ./cctv3.jpg} is executed"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image


def histogram_equalisation(img: np.ndarray) -> np.ndarray:
    histogram = np.bincount(img.flatten(), minlength=256)

    normalised_histogram = histogram / np.sum(histogram)
    cumulative_histogram = np.cumsum(normalised_histogram)

    pixel_intensity_map = np.floor(255 * cumulative_histogram).astype(np.uint8)

    flattened_image_array = list(img.flatten())
    equalised_image_array = [pixel_intensity_map[p] for p in flattened_image_array]

    output_image = np.reshape(np.asarray(equalised_image_array), img.shape)

    return output_image.astype(np.uint8)


def gamma_correction(img, gamma):
    return cv2.LUT(
        img,
        np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]
        ).astype(np.uint8),
    )


def filter_and_enhance(img: np.ndarray) -> np.ndarray:
    return gamma_correction(histogram_equalisation(img), 0.7)


def save_image(img: np.ndarray, filename: str):
    Image.fromarray(img).save(filename)


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "filepath",
            type=str,
            help="Image filepath",
        )
        args = parser.parse_args()

        if not os.path.exists(args.filepath):
            print("Please enter valid file path for input cctv image")
            return -1

        filename = os.path.basename(args.filepath)

        img = np.array(Image.open(args.filepath))

        img = filter_and_enhance(img)

        save_image(img, f"enchanced-{filename}")

        print(f"CCTV image enhanced and saved successfully as enhanced-{filename}")
        return 0

    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    sys.exit(main())
