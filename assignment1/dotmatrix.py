"""
Usage:
    pip install numpy pillow
    python dotmatrix.py n_d
    , where n_d \in {00,01,..,99}
"""
__author__ = "Gurbaaz Singh Nandra"
__rollno__ = 190349
__filename__ = "dotmatrix.py"
__description__ = """dotmatrix.py, which accepts command line docking station number 
nd ∈ {00, 01, ..., 99} and directly stores the dotmatrix.jpg file at current location"""

import sys
import argparse
from typing import Tuple
import numpy as np
from PIL import Image

digits_pattern = {
    0: [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14],
    1: [2, 5, 8, 11, 14],
    2: [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 14],
    3: [0, 1, 2, 5, 6, 7, 8, 11, 12, 13, 14],
    4: [0, 2, 3, 5, 6, 7, 8, 11, 14],
    5: [0, 1, 2, 3, 6, 7, 8, 11, 12, 13, 14],
    6: [0, 1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14],
    7: [0, 1, 2, 5, 8, 11, 14],
    8: [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14],
    9: [0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14],
}

circles_centers = []

for i in [30, 90, 150, 210, 270]:
    for j in [65, 125, 185]:
        circles_centers.append((i, j))


def shift_offset(cc: Tuple) -> Tuple:
    x, y = cc
    return (x, y + 250)


def draw_circle(A: np.ndarray, center: Tuple, radius: int) -> np.ndarray:
    x0, y0 = center

    for i in range(x0 - radius, x0 + radius):
        for j in range(y0 - radius, y0 + radius):
            if (i - x0) ** 2 + (j - y0) ** 2 <= radius**2:
                A[(i, j)] = 1

    return A


def draw_digits(A: np.ndarray, pattern: str) -> np.ndarray:
    global circles_centers
    global digits_pattern

    for i, p in enumerate(pattern):
        for dp in digits_pattern[int(p)]:
            circle_center = circles_centers[dp]

            if i == 1:
                circle_center = shift_offset(circle_center)

            A = draw_circle(A, circle_center, 25)

    return A


def save_image(A: np.ndarray, filename: str):
    Image.fromarray(A * 255).convert("L").save(filename)


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "n_d",
            type=str,
            help="Docking station number nd ∈ {00, 01, ..., 99}",
        )
        args = parser.parse_args()

        if not len(args.n_d) == 2 or not args.n_d.isnumeric() or args.n_d[0] == "-":
            print(
                "Please enter two digit docking station number as nd ∈ {00, 01, ..., 99}"
            )
            return -1

        A = np.zeros((300, 500))

        A = draw_digits(A, args.n_d)

        save_image(A, "dotmatrix.jpg")

        print("Dotmatrix image generated and saved successfully as dotmatrix.jpg")
        return 0

    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    sys.exit(main())
