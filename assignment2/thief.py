"""
Usage:
    pip install opencv-python numpy
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
from typing import Tuple
import numpy as np
import cv2


def filter_and_enhance(img):
    img = cv2.equalizeHist(img)    
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
            print("Please enter valid file path for input cctv image")
            return -1
        
        filename = os.path.basename(args.filepath)

        img = cv2.imread(args.filepath, 0)

        img = filter_and_enhance(img)

        save_image(img, f"enchanced-{filename}")

        print(f"Image enhanced and saved successfully as enhanced-{filename}")
        return 0

    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    sys.exit(main())
