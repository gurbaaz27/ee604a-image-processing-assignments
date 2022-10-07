"""
Usage:
    pip install opencv-python numpy
    python gutter.py path-to-image
"""
__author__ = "Gurbaaz Singh Nandra"
__rollno__ = 190349
__filename__ = "gutter.py"
__description__ = """python program file gutter.py, which accepts command line file path for input image
and directly stores the cleaned-gutter.jpg file at current location where the linux command
{$python gutter.py ./gutters1.jpg} is being executed"""

import os
import sys
import argparse
from typing import Tuple
import cv2
import numpy as np


def clean_gutter(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

    shadowremov = cv2.merge(result_norm_planes)
    
    return shadowremov


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
            print("Please enter valid file path for input gutter image")
            return -1
        
        img = cv2.imread(args.filepath, -1)

        img = clean_gutter(img)

        save_image(img, "cleaned-gutter.jpg")

        print("Gutter image cleaned and saved successfully as cleaned-gutter.jpg")
        return 0

    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    sys.exit(main())
