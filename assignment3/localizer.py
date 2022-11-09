"""
Usage:
    python localizer.py path-to-image
"""
__author__ = "Gurbaaz Singh Nandra"
__rollno__ = 190349
__filename__ = "localizer.py"
__description__ = """python program localizer.py, which accepts com-
mand line file path for input image and directly out-
puts the location number (either 1, 2, or 3) after the
command {$python localizer.py ./location2.jpg} is executed."""

import os
import sys
import argparse
import cv2
import numpy as np
from typing import Dict


def average_image_pixels(img) -> float:
    return np.average(img)


def localize(img) -> int:
    blue_avg: float = average_image_pixels(img[:, :, 0])
    green_avg: float = average_image_pixels(img[:, :, 1])
    red_avg: float = average_image_pixels(img[:, :, 2])

    net_avg: float = blue_avg + green_avg + red_avg

    if abs(3 * green_avg - net_avg) > 28:
        return 2

    if abs(blue_avg - red_avg) > 20:
        return 3

    return 1


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
            print("Please enter valid file path for input image")
            return -1

        img = cv2.imread(args.filepath)

        output_location: int = localize(img)

        location_dict: Dict = {1: "Building", 2: "Grass", 3: "Road"}

        print(f"Location: {output_location} ({location_dict[output_location]})")
        return 0

    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    sys.exit(main())
