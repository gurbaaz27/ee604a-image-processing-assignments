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
import cv2
import numpy as np


def plot_hough_transform(img, peak_indices, thetas, rhos):
    cyan_color = (200, 200, 0)
    sz = 5000

    for i in range(len(peak_indices)):
        rho, theta = rhos[peak_indices[i][0]], thetas[peak_indices[i][1]]

        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho

        p0 = (int(x0 + sz * (-b)), int(y0 + sz * (a)))
        p1 = (int(x0 - sz * (-b)), int(y0 - sz * (a)))

        cv2.line(img, p0, p1, cyan_color, 2)

    return img


def find_peak_indices(accum, thetas, rhos):
    peak_indices = []

    threshold = 110
    neighbourhood_size = 40

    hough_transform = np.copy(accum)

    while True:
        id = np.argmax(hough_transform)
        ht_id = np.unravel_index(id, hough_transform.shape)

        # Exit condition
        if hough_transform[ht_id] < threshold:
            break

        peak_indices.append(ht_id)
        yid, xid = ht_id

        if xid - (neighbourhood_size / 2) < 0:
            min_x = 0
        else:
            min_x = int(xid - (neighbourhood_size / 2))

        if xid + (neighbourhood_size / 2) + 1 > accum.shape[1]:
            max_x = accum.shape[1]
        else:
            max_x = int(xid + (neighbourhood_size / 2) + 1)

        if yid - (neighbourhood_size / 2) < 0:
            min_y = 0
        else:
            min_y = int(yid - (neighbourhood_size / 2))

        if yid + (neighbourhood_size / 2) + 1 > accum.shape[0]:
            max_y = accum.shape[0]
        else:
            max_y = int(yid + (neighbourhood_size / 2) + 1)

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                hough_transform[y, x] = 0

                if x == min_x or x == (max_x - 1):
                    accum[y, x] = 255

                if y == min_y or y == (max_y - 1):
                    accum[y, x] = 255

    return peak_indices, thetas, rhos


def hough_transform(img):
    width, height = img.shape
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    max_length = int(np.ceil(np.sqrt(width * width + height * height)))
    rhos = np.arange(-max_length, max_length + 1, 1)

    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    num_thetas = len(thetas)
    num_rhos = len(rhos)

    accum = np.zeros((num_rhos, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]

        for theta in range(num_thetas):
            rho = int(round(x * cos_thetas[theta] + y * sin_thetas[theta])) + max_length
            accum[rho, theta] += 1

    return accum, thetas, rhos


def tile_lines(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges_image = cv2.Canny(grey_img, 120, 200)
    peak_indices, thetas, rhos = find_peak_indices(*hough_transform(edges_image))
    return plot_hough_transform(img, peak_indices, thetas, rhos)


def save_image(img: np.ndarray, filename: str):
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

        img = cv2.imread(args.filepath)

        img = tile_lines(img)

        save_image(img, f"robolin-{filename}")

        print(f"Tile lines detected and saved successfully as robolin-{filename}")
        return 0

    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    sys.exit(main())
