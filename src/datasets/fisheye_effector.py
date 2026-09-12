import argparse
import io
import os
import time
from math import sqrt

import numpy as np
from PIL import Image


class FisheyeEffector:
    """
    Fisheye transform used by the original paper/repository.

    The geometric mapping is kept identical to the original implementation.
    The only change is how the fixed pixel mapping is represented/applied:
    instead of constructing a huge sparse PyTorch matrix and doing sparse.mm
    for every image, we precompute the equivalent source-pixel lookup table
    and use NumPy indexing.
    """

    def __init__(self, height=720, width=1280, distortion=0.5):
        self.float_height, self.float_width = float(height), float(width)
        self.height, self.width = height, width
        self.setDistortion(distortion=distortion)

    def setDistortion(self, distortion=0.5):
        self.distortion = distortion
        self.crop = distortion > 0
        self.left, self.upper, self.right, self.lower = 0, 0, self.width, self.height
        self.key_coordinates = []

        # Vectorized equivalent of the original per-pixel mapping calculation.
        # We deliberately preserve the original flattening rule:
        # the original code accepted a mapping when j = org_h * width + org_w
        # was inside range(num_pixels), even when org_h/org_w individually
        # landed outside the image.
        h = np.arange(self.height, dtype=np.float64)[:, None]
        w = np.arange(self.width, dtype=np.float64)[None, :]

        norm_h = (2 * h - self.height) / self.height
        norm_w = (2 * w - self.width) / self.width
        diagonal = norm_h - norm_w == 0

        norm_h = norm_h * self.float_height / self.float_width

        radius = np.sqrt(norm_h ** 2 + norm_w ** 2)
        denominator = 1 - distortion * (radius ** 2)
        valid_denominator = denominator != 0

        shape = (self.height, self.width)
        norm_h_full = np.broadcast_to(norm_h, shape)
        norm_w_full = np.broadcast_to(norm_w, shape)
        denominator_full = np.broadcast_to(denominator, shape)
        valid_denominator_full = np.broadcast_to(valid_denominator, shape)

        org_norm_h = np.divide(
            norm_h_full,
            denominator_full,
            out=np.zeros(shape, dtype=np.float64),
            where=valid_denominator_full,
        )
        org_norm_w = np.divide(
            norm_w_full,
            denominator_full,
            out=np.zeros(shape, dtype=np.float64),
            where=valid_denominator_full,
        )

        org_norm_h = org_norm_h * self.float_width / self.float_height

        # np.trunc matches the behavior of int() for these floating-point values.
        org_h = np.trunc(
            (org_norm_h * self.height + self.height) / 2
        ).astype(np.int64)
        org_w = np.trunc(
            (org_norm_w * self.width + self.width) / 2
        ).astype(np.int64)

        num_pixels = self.width * self.height
        source_index = org_h * self.width + org_w

        valid = (
            valid_denominator_full
            & (source_index >= 0)
            & (source_index < num_pixels)
        )

        # Invalid rows in the original sparse matrix are all zeros.
        self.source_index = np.where(valid, source_index, 0).astype(np.int32).reshape(-1)
        self.valid_mask = valid.reshape(-1)

        # Preserve the original crop-boundary behavior, but only iterate over
        # diagonal candidates instead of every image pixel.
        diagonal_valid = diagonal & valid
        diagonal_y, diagonal_x = np.nonzero(diagonal_valid)
        for y, x in zip(diagonal_y.tolist(), diagonal_x.tolist()):
            if self.left == 0 and self.upper == 0:
                self.left, self.upper = x, y
            self.right, self.lower = x, y

        # calculate key coordinates
        candidate_norms = [(0.2, 0.1), (0.2, 0), (0.2, -0.1)]
        expansion = 1.0
        if self.crop:
            expansion = self.float_width / float(self.right - self.left)

        for norm_x, norm_y in candidate_norms:
            radius = sqrt(norm_x ** 2 + norm_y ** 2)
            dst_x, dst_y = calc_points_of_distorted_image(
                norm_x, norm_y, radius, distortion
            )
            self.key_coordinates.append(
                (dst_x * expansion, dst_y * expansion)
            )

        print(
            "FisheyeEffector was initialized with distortion = {}".format(
                distortion
            )
        )

    def getKeyCoordinates(self):
        return self.key_coordinates

    def getDistortion(self):
        return self.distortion

    def apply(self, image_bytes):
        return self.calcImage(Image.open(io.BytesIO(image_bytes)))

    def __call__(self, image):
        # The original __call__ eventually returned a PIL image.  Apply the
        # exact same mapping without the intermediate PNG serialization.
        return Image.fromarray(self._transform_array(image))

    def _transform_array(self, image):
        image = np.array(image)

        # padding
        image = padding(image, height=self.height, width=self.width)

        org_dtype = image.dtype
        org_shape = image.shape

        # Equivalent to the original sparse matrix multiplication:
        # every output pixel receives exactly one source pixel value.
        flat_image = image.reshape(-1, image.shape[2])
        fish_image = flat_image[self.source_index]

        if not np.all(self.valid_mask):
            fish_image = fish_image.copy()
            fish_image[~self.valid_mask] = 0

        fish_image = fish_image.astype(org_dtype, copy=False).reshape(org_shape)

        fish_image = Image.fromarray(fish_image)
        if self.crop:
            fish_image = fish_image.crop(
                (self.left, self.upper, self.right, self.lower)
            )
        fish_image = fish_image.resize(
            (self.width, self.height),
            Image.Resampling.LANCZOS,
        )

        return np.asarray(fish_image)

    def calcImage(self, image):
        fish_image = Image.fromarray(self._transform_array(image))

        fish_image_bytes = io.BytesIO()
        fish_image.save(fish_image_bytes, "png")

        return fish_image_bytes.getvalue()


def calc_points_of_original_image(x, y, r, distortion):
    if distortion > 1:
        distortion = 1
    elif distortion < -1:
        distortion = -1

    if 1 - distortion * (r ** 2) == 0:
        return x, y

    return (
        x / (1 - distortion * (r ** 2)),
        y / (1 - distortion * (r ** 2)),
    )


def calc_points_of_distorted_image(x, y, r, distortion):
    if distortion > 1:
        distortion = 1
    elif distortion < -1:
        distortion = -1

    if distortion == 0 or x == 0:
        return x, y

    a = distortion * x * (1 + y ** 2 / x ** 2)
    c = -x

    new_x = (-1 + sqrt(1 - 4 * a * c)) / (2 * a)
    new_y = (y / x) * new_x

    return new_x, new_y


def padding(image, height=720, width=1280):
    src_height, src_width, _ = image.shape
    if src_height < height and src_width < width:
        pad_h = int((height - src_height) / 2)
        pad_w = int((width - src_width) / 2)
        image = np.pad(
            image,
            [(pad_h, pad_h), (pad_w, pad_w), (0, 0)],
            "constant",
        )
    return image


def execDir(effector, path):
    for file in os.listdir(path):
        joined_path = os.path.join(path, file)

        if os.path.isfile(joined_path):
            _, ext = os.path.splitext(joined_path)
            output_path = joined_path.replace(ext, "_dis.png")
            execFile(effector, joined_path, output_path=output_path)

        else:
            execDir(effector, joined_path)


def execFile(effector, input_path, output_path="output.png"):
    if input_path.endswith("_dis.png"):
        print("skip", input_path)
        return

    if os.path.exists(output_path):
        print("skip", input_path)
        return

    with open(input_path, "rb") as image_bin:
        print(input_path)
        output = open(output_path, "wb")
        output.write(effector.apply(image_bin.read()))
        output.close()


def resizeAndApply(
    effector,
    width,
    height,
    input_path,
    output_path="output.png",
):
    print(input_path)
    image = Image.open(input_path)
    image = image.resize((width, height))
    image = effector(image)
    image.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        help="path to the input image or directory",
    )
    parser.add_argument(
        "-d",
        "--distortion",
        type=float,
        default=0.1,
        help="amount of distortion between -1 to 1 (0.1 as default)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="input image width (1280 as default)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="input image height (720 as default)",
    )

    args = parser.parse_args()
    input_path = args.input
    distortion = args.distortion
    width = args.width
    height = args.height

    if not os.path.exists(input_path):
        print("No such file or directory: {}".format(input_path))
        exit(1)

    start = time.time()
    effector = FisheyeEffector(
        height=height,
        width=width,
        distortion=distortion,
    )
    end = time.time()

    print("{} sec had been spent for Initialize.".format(end - start))

    if os.path.isfile(input_path):
        start = time.time()
        resizeAndApply(effector, width, height, input_path)
        end = time.time()

    else:
        execDir(effector, input_path)
