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
import cv2
import numpy as np


class GaussianBlur:
    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.dim = kernel_size // 2

    def prepare(self):
        self.gaussian_kernel = np.zeros(
            (self.kernel_size, self.kernel_size), np.float32
        )

        for x in range(-self.dim, self.dim + 1):
            for y in range(-self.dim, self.dim + 1):
                x1 = 2 * np.pi * (self.sigma**2)
                x2 = np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
                self.gaussian_kernel[
                    x + self.dim, y + self.dim
                ] = (1 / x1) * x2

    def convolve(self, img, kernel):
        image_pad = np.pad(
            img,
            pad_width=((self.dim, self.dim), (self.dim, self.dim)),
            mode="constant",
            constant_values=0,
        ).astype(np.float32)

        image_conv = np.zeros(image_pad.shape)

        for i in range(self.dim, image_pad.shape[0] - self.dim):
            for j in range(self.dim, image_pad.shape[1] - self.dim):
                x = image_pad[
                    i - self.dim : i - self.dim + self.kernel_size, j - self.dim : j - self.dim + self.kernel_size
                ]
                x = x.flatten() * kernel.flatten()
                image_conv[i][j] = x.sum()

        return image_conv[self.dim:-self.dim, self.dim:-self.dim].astype(np.uint8)

    def apply(self, img):
        return self.convolve(img, self.gaussian_kernel)


def normalised_image(img):
    return cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC1,
    )


def gamma_correction(img, gamma):
    return cv2.LUT(
        img,
        np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]
        ).astype(np.uint8),
    )


def clean_gutter(img):
    cleaned_img = []

    img = gamma_correction(img, 1.5)

    gaussian_blur = GaussianBlur(kernel_size=15, sigma=3)
    gaussian_blur.prepare()

    for axis in cv2.split(img):
        processed_img = cv2.dilate(axis, np.ones((6, 6), np.uint8))
        blurred_image = gaussian_blur.apply(processed_img)
        cleaned_img.append(normalised_image(255 - cv2.absdiff(blurred_image, axis)))

    return gamma_correction(cv2.merge(cleaned_img), 0.5)


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

        img = cv2.imread(args.filepath)

        img = clean_gutter(img)

        save_image(img, "cleaned-gutter.jpg")

        print("Gutter image cleaned and saved successfully as cleaned-gutter.jpg")
        return 0

    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    sys.exit(main())
