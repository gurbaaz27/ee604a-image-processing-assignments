"""
Usage:
    pip install numpy pillow
    python jigsolver.py ./jigsolver.jpg
"""
__author__ = "Gurbaaz Singh Nandra"
__rollno__ = 190349
__filename__ = "jigsolver.py"
__description__ = """jigsolver.py, which accepts command line file path for input jigsaw.jpg and 
directly stores the jigsolved.jpg file at current location where the command is being executed"""

import os
import sys
import argparse
from typing import Tuple
import numpy as np
from PIL import Image


def flip_image(A: np.ndarray, xy: Tuple, hw: Tuple, horizontal=True) -> np.ndarray:
    x, y = xy
    h, w = hw

    if horizontal:
        for i in range(x, x + h):
            itr = 0
            for j in range(y, y + (w // 2)):
                temp = np.copy(A[(i, j)])
                A[(i, j)] = A[(i, y + w - 1 - itr)]
                A[(i, y + w - 1 - itr)] = temp
                itr += 1

    else:
        for i in range(y, y + w):
            itr = 0
            for j in range(x, x + (h // 2)):
                temp = np.copy(A[(j, i)])
                A[(j, i)] = A[(x + h - 1 - itr, i)]
                A[(x + h - 1 - itr, i)] = temp
                itr += 1

    return A


def swap_image(
    A: np.ndarray, xy1: Tuple, hw1: Tuple, xy2: Tuple, hw2: Tuple
) -> np.ndarray:
    x1, y1 = xy1
    h1, w1 = hw1
    x2, y2 = xy2
    h2, w2 = hw2

    for i in range(min(h1, h2)):
        for j in range(min(w1, w2)):
            temp = np.copy(A[(x1 + i, y1 + j)])
            A[(x1 + i, y1 + j)] = A[(x2 + i, y2 + j)]
            A[(x2 + i, y2 + j)] = temp

    return A


def mirror_padding(A: np.ndarray, xy: Tuple, hw: Tuple) -> np.ndarray:
    x, y = xy
    h, w = hw

    for i in range(h // 2):
        for j in range(w):
            A[(x + i, y + j)] = A[(x - i - 1, y + j)]

    for i in range(h // 2):
        for j in range(w):
            A[(x + h - i - 1, y + j)] = A[(x + h + i, y + j)]

#    for j in range(w):
 #       A[(x + h // 2, y + j)] = ( A[(x + h // 2 - 1, y + j)] + A[(x + h // 2 + 1, y + j)] ) // 2

    return A


def solve_jigsaw(A: np.ndarray) -> np.ndarray:
    A = flip_image(A, (370, 370), (51, 427), False)
    A = flip_image(A, (150, 515), (179, 185))
    A = flip_image(A, (200, 0), (210, 190), False)
    A = swap_image(A, (0, 0), (200, 190), (200, 0), (210, 190))
    
    for i in range(200, 400):
        for j in range(190):
            pxl = A[(i, j)]
            pxl[1], pxl[2] = pxl[2], pxl[1]

    A = mirror_padding(A, (400, 0), (10, 190))

    return A


def save_image(A: np.ndarray, filename: str):
    Image.fromarray(A).save(filename)


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "filepath",
            type=str,
            help="file path for input jigsaw.jpg",
        )
        args = parser.parse_args()

        if not os.path.exists(args.filepath):
            print("Please enter valid file path for input jigsaw.jpg")
            return -1

        A = np.array(Image.open(args.filepath))

        A = solve_jigsaw(A)

        save_image(A, "jigsolved.jpg")

        print("Jigsaw image solved and saved successfully as jigsolved.jpg")
        return 0

    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    sys.exit(main())
