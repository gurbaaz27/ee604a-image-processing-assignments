"""
Usage:
    pip install opencv-python numpy
    python robolin.py path-to-image
"""
__author__ = "Gurbaaz Singh Nandra"
__rollno__ = 190349
__filename__ = "robolin.py"
__description__ = """python program file
robolin.py, which accepts command line file path for
an input image and directly stores robolin-tiles3.jpg
file when the linux command {$python robolin.py ./tiles3.jpg} is being executed"""

import os
import sys
import argparse
from typing import Tuple
import cv2
import numpy as np


def tile_lines(img):
    
    return img


def save_image(img, filename: str):
    cv2.imwrite(filename, img)


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
            print("Please enter valid file path for input tiles image")
            return -1

        filename = os.path.basename(args.filepath)
        
        img = cv2.imread(args.filepath, -1)

        img = tile_lines(img)

        save_image(img, f"robolin-{filename}")

        print(f"Tile lines detected and saved successfully as robolin-{filename}")
        return 0

    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    sys.exit(main())
