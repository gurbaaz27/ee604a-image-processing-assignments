"""
Usage:
    python colorizer.py path-to-Y.jpg path-to-Cb4.jpg path-to-Cr4.jpg
"""
__author__ = "Gurbaaz Singh Nandra"
__rollno__ = 190349
__filename__ = "colorizer.py"
__description__ = """python program colorizer.py, which accepts command line file path for all 3 input
images (viz. Y channel original shape image, 4 times decimated Cb and Cr channel images) and
directly stores the colorful output image flyingelephant.jpg file at current location where the
command {$python colorizer.py ./Y.jpg ./Cb4.jpg ./Cr4.jpg} is being executed"""

import os
import sys
import argparse
import cv2
import numpy as np


class BilateralUpsample:
    def __init__(
        self,
        Y: np.ndarray,
        Cb: np.ndarray,
        Cr: np.ndarray,
        kernel_size: int,
        sigma: int,
        sigma_r: int,
        sampling_rate: int,
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.sigma_r = sigma_r
        self.sampling_rate = sampling_rate
        self.dim = kernel_size // 2
        self.Y = Y
        self.Cb = Cb
        self.Cr = Cr
        self.padder = ((self.dim, self.dim), (self.dim, self.dim))
        self.initialize_output_images()
        self.set_image_size()

    def set_image_size(self):
        self.height, self.width = np.shape(self.Y)

    def initialize_output_images(self):
        self.output_Cb = np.zeros((np.shape(self.Y)))
        self.output_Cr = np.zeros((np.shape(self.Y)))

    def prepare(self):
        self.gaussian_kernel = np.zeros(
            (self.kernel_size, self.kernel_size), np.float32
        )

        for x in range(-self.dim, self.dim + 1):
            for y in range(-self.dim, self.dim + 1):
                x1 = 2 * np.pi * (self.sigma**2)
                x2 = np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
                self.gaussian_kernel[x + self.dim, y + self.dim] = (1 / x1) * x2

    def pad_image(self, image: np.ndarray):
        return np.pad(image, self.padder)

    def apply_filter(self):
        padded_Y = self.pad_image(self.Y)
        padded_Cb = self.pad_image(self.Cb)
        padded_Cr = self.pad_image(self.Cr)

        for i in range(self.height):
            for j in range(self.width):
                cx, cy = (2 * i + self.kernel_size) // 2, (
                    2 * j + self.kernel_size
                ) // 2
                range_kernel = np.exp(
                    -0.5
                    * (
                        (
                            padded_Y[i : i + self.kernel_size, j : j + self.kernel_size]
                            - padded_Y[cx, cy]
                        )
                        / self.sigma_r
                    )
                    ** 2
                )
                kernel = range_kernel * self.gaussian_kernel
                normalised_kernel = kernel / np.sum(kernel)
                self.output_Cb[i, j] = np.sum(
                    padded_Cb[i : i + self.kernel_size, j : j + self.kernel_size]
                    * normalised_kernel
                )
                self.output_Cr[i, j] = np.sum(
                    padded_Cr[i : i + self.kernel_size, j : j + self.kernel_size]
                    * normalised_kernel
                )


def rescale_image(img: np.ndarray, base_image: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (np.shape(base_image)[1], np.shape(base_image)[0])).astype(
        float
    )


def colorize(Y: np.ndarray, Cb4: np.ndarray, Cr4: np.ndarray) -> np.ndarray:
    bilateral_upsample = BilateralUpsample(
        Y, Cb4, Cr4, kernel_size=5, sigma=5, sigma_r=5, sampling_rate=4
    )
    bilateral_upsample.prepare()
    bilateral_upsample.apply_filter()

    final_image = (
        np.array([Y, bilateral_upsample.output_Cr, bilateral_upsample.output_Cb])
        .astype(np.uint8)
        .transpose(1, 2, 0)
    )

    return final_image


def read_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, 0)
    return np.asarray(img).astype(float)


def save_image(img, filename: str):
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(filename, img)


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "Y",
            type=str,
            help="Y.jpg filepath",
        )
        parser.add_argument(
            "Cb4",
            type=str,
            help="Cb4.jpg filepath",
        )
        parser.add_argument(
            "Cr4",
            type=str,
            help="Cr4.jpg filepath",
        )
        args = parser.parse_args()

        if not os.path.exists(args.Y):
            print("Please enter valid file path for Y.jpg")
            return -1

        if not os.path.exists(args.Cb4):
            print("Please enter valid file path for Cb4.jpg")
            return -1

        if not os.path.exists(args.Cr4):
            print("Please enter valid file path for Cr4.jpg")
            return -1

        Y = read_image(args.Y)
        Cb4 = rescale_image(read_image(args.Cb4), Y)
        Cr4 = rescale_image(read_image(args.Cr4), Y)

        img = colorize(Y, Cb4, Cr4)

        save_image(img, "flyingelephant.jpg")

        print(f"Image colorized and saved successfully as flyingelephant.jpg")
        return 0

    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    sys.exit(main())
