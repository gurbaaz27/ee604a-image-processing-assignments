"""
Usage:
    python denoiser.py path-to-image
"""
__author__ = "Gurbaaz Singh Nandra"
__rollno__ = 190349
__filename__ = "denoiser.py"
__description__ = """python program denoiser.py, which accepts com-
mand line file path for input image and directly stores
the output denoised.jpg file at current location where
the command {$python denoiser.py ./noisy1.jpg} is
being executed."""

import os
import sys
import argparse
import cv2
import numpy as np


class BilateralFilter:
    def __init__(self, image: np.ndarray, kernel_size: int, sigma: int, sigma_r: int):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.sigma_r = sigma_r
        self.dim = kernel_size // 2
        self.image = image.transpose(2, 0, 1)
        self.output_image = np.zeros(np.shape(image))
        self.set_image_size(image)

    def set_image_size(self, image: np.ndarray):
        self.height, self.width, self.channels = np.shape(image)

    def prepare(self):
        self.gaussian_kernel = np.zeros(
            (self.kernel_size, self.kernel_size), np.float32
        )

        for x in range(-self.dim, self.dim + 1):
            for y in range(-self.dim, self.dim + 1):
                x1 = 2 * np.pi * (self.sigma**2)
                x2 = np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
                self.gaussian_kernel[x + self.dim, y + self.dim] = (1 / x1) * x2

    def pad_image(self):
        padder = ((self.dim, self.dim),(self.dim, self.dim))

        R = np.pad(self.image[0], padder)
        G = np.pad(self.image[1], padder)
        B = np.pad(self.image[2], padder)

        self.padded_image = np.array([R, G, B]).transpose(1, 2, 0)

    def apply_filter(self):
        self.pad_image()

        for i in range(self.height):
            for j in range(self.width):
                for k in range(self.channels):
                    cx, cy = (2 * i + self.kernel_size) // 2, (
                        2 * j + self.kernel_size
                    ) // 2
                    range_kernel = np.exp(
                        -(
                            (
                                self.padded_image[
                                    i : i + self.kernel_size,
                                    j : j + self.kernel_size,
                                    k,
                                ]
                                - self.padded_image[cx, cy, k]
                            )
                            ** 2
                        )
                        / (2 * (self.sigma_r**2))
                    )
                    kernel = range_kernel * self.gaussian_kernel
                    normalised_kernel = kernel / np.sum(kernel)
                    self.output_image[i, j, k] = np.sum(
                        self.padded_image[
                            i : i + self.kernel_size, j : j + self.kernel_size, k
                        ]
                        * normalised_kernel
                    )


def denoise_image(img):
    img = np.array(img).astype(float)

    bilateral_filter = BilateralFilter(img, kernel_size=13, sigma=9, sigma_r=30)
    bilateral_filter.prepare()
    bilateral_filter.apply_filter()

    return bilateral_filter.output_image.astype(np.uint8)


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
            print("Please enter valid file path for input image")
            return -1

        img = cv2.imread(args.filepath)

        img = denoise_image(img)

        save_image(img, "denoised.jpg")

        print(f"Image denoised and saved successfully as denoised.jpg")
        return 0

    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    sys.exit(main())
