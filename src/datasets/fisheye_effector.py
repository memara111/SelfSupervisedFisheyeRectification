import numpy as np
import torch

from core.division_model import (
    MAX_DISTORTION,
    clamp_distortion,
    distorted_to_original,
    original_to_distorted,
    valid_radius,
)

#: Key points the network has to bring back into agreement: three points that are
#: vertically aligned and equally spaced in the rectilinear image. Expressed in
#: isotropic normalised units (see ``core.division_model``).
KEY_CANDIDATES = ((0.2, 0.1), (0.2, 0.0), (0.2, -0.1))


class FisheyeEffector:
    """Renders the single parameter division model as a sparse resampling matrix.

    ``backward=False``  rectilinear -> fisheye (what the datasets use to make data).

    ``backward=True``   fisheye -> rectilinear (rectification; what ``test.py`` and
    ``predict.py`` need). The two directions are exact inverses of each other because
    both are built from :mod:`core.division_model`.

    The crop that keeps a positive-distortion render free of invalid border pixels is
    folded into the sampling map instead of being done with ``Image.crop`` +
    ``Image.resize`` afterwards. That keeps the geometry exact (one resampling, no
    double interpolation), makes ``expansion`` known analytically, and removes the
    per-pixel Python loop that made construction cost O(H*W) iterations.
    """

    def __init__(
        self,
        height=720,
        width=1280,
        distortion=0.5,
        backward=False,
        crop=True,
        device="cpu",
        verbose=False,
    ):
        self.height, self.width = int(height), int(width)
        self.device = device
        self.backward = bool(backward)
        self.crop = bool(crop)
        self.verbose = verbose
        self.setDistortion(distortion=distortion)

    # ------------------------------------------------------------------ geometry
    @property
    def aspect(self):
        """y-scale that turns height-normalised units into isotropic (width) units."""
        return self.height / self.width

    def setDistortion(self, distortion=0.5):
        self.distortion = float(np.clip(distortion, -MAX_DISTORTION, MAX_DISTORTION))
        self.k = torch.tensor([self.distortion], dtype=torch.float64)
        self._build()
        if self.verbose:
            print(
                "FisheyeEffector was initialized with distortion = {} (backward={})".format(
                    self.distortion, self.backward
                )
            )

    def _grid(self):
        """Isotropic normalised coordinates of every pixel of the W x H frame."""
        cols = np.arange(self.width, dtype=np.float64)
        rows = np.arange(self.height, dtype=np.float64)
        x = (2.0 * cols - self.width) / self.width
        y = ((2.0 * rows - self.height) / self.height) * self.aspect
        gx, gy = np.meshgrid(x, y)
        return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)

    def frameRadius(self):
        """Isotropic radius of a frame corner, in width-normalised units."""
        return float(np.sqrt(1.0 + self.aspect ** 2))

    def _computeExpansion(self):
        """Magnification applied by cropping the render to its valid region.

        A positive ``k`` pushes content outwards, so the frame corners leave the input.
        The render is therefore cropped to the largest centred box that keeps the
        frame's *aspect ratio* (otherwise a single scale factor would stretch one axis
        more than the other) and rescaled. Its half-diagonal is exactly the largest
        radius that stays valid, hence ``frame_radius / valid_radius``.
        """
        r = valid_radius(self.distortion)
        if r is None or not self.crop or r <= 0.0:
            return 1.0
        return max(1.0, self.frameRadius() / r)

    def _build(self):
        """Sampling map + bilinear weights, plus the key coordinate labels."""
        self.expansion = self._computeExpansion()
        self._buildTaps()
        self._buildKeyCoordinates()

    def _sourceCoords(self):
        """Where each output pixel samples from, in *input image* normalised units."""
        grid = self._grid()  # output frame
        k = self.k
        if self.backward:
            # rectify: output is rectilinear, sample the fisheye where its content is
            # (the fisheye frame we were given is the magnified/cropped one)
            coords = original_to_distorted(torch.from_numpy(grid), k).numpy()
            coords *= self.expansion
        else:
            # render: output is the (cropped, magnified) fisheye frame
            coords = distorted_to_original(torch.from_numpy(grid / self.expansion), k).numpy()
        # isotropic -> per axis normalised: y has to leave the isotropic convention
        coords[:, 1] /= self.aspect if self.aspect else 1.0
        return coords

    def _buildTaps(self):
        """Bilinear taps (source index + weight) for every output pixel.

        Kept as plain index/weight tensors rather than a sparse matrix: a ``(3P, 3P)``
        sparse tensor with four taps per row costs ~94 MB per effector at 512 x 256, and
        a dataset holds one effector per distortion fragment. Gathering costs ~8 MB and
        is faster besides.
        """
        num_pixels = self.width * self.height
        coords = self._sourceCoords()

        # normalised -> pixel index. The "corner" convention ((x + 1) * size / 2, no
        # half pixel shift) is deliberate: it makes k = 0 sample exactly its own pixel,
        # so the identity case is bit exact and the frame border is not darkened.
        fx = (coords[:, 0] + 1.0) * 0.5 * self.width
        fy = (coords[:, 1] + 1.0) * 0.5 * self.height
        self.source_pixels = np.stack([fx, fy], axis=-1)
        x0 = np.floor(fx)
        y0 = np.floor(fy)
        wx = (fx - x0)[:, None]
        wy = (fy - y0)[:, None]

        indices, weights = [], []
        for dy, ay in ((0.0, 1.0 - wy), (1.0, wy)):
            for dx, ax in ((0.0, 1.0 - wx), (1.0, wx)):
                sx = (x0 + dx).astype(np.int64)
                sy = (y0 + dy).astype(np.int64)
                # per axis bounds check: an out of frame sample contributes nothing, and
                # is never folded back onto another row (the old code checked the
                # *linear* index only, so (row -1, col W+6) aliased onto a valid pixel)
                inside = (sx >= 0) & (sx < self.width) & (sy >= 0) & (sy < self.height)
                weight = (ay * ax).reshape(-1) * inside
                indices.append((sy * self.width + sx).clip(0, num_pixels - 1))
                weights.append(weight)

        self.tapIndex = torch.tensor(np.stack(indices), dtype=torch.long)
        self.tapWeight = torch.tensor(np.stack(weights), dtype=torch.float32)[:, :, None]

    def _buildKeyCoordinates(self):
        """Positions of the key points in the image this effector *delivers*."""
        cand = torch.tensor(np.array(KEY_CANDIDATES, dtype=np.float64))
        if self.backward:
            # labels are only used by the distortion direction; keep the attribute
            # defined so the object is always usable
            self.key_coordinates = cand.numpy()
            return
        pts = original_to_distorted(cand, self.k) * self.expansion
        self.key_coordinates = pts.numpy()

    # ------------------------------------------------------------------- access
    def getKeyCoordinates(self):
        """``(3, 2)`` float array of key points in the delivered image, isotropic units."""
        return self.key_coordinates.copy()

    def getExpansion(self):
        """Magnification of the delivered frame relative to the uncropped render."""
        return float(self.expansion)

    def getDistortion(self):
        return self.distortion

    def sourcePixels(self):
        """``(H*W, 2)`` float pixel coordinates each output pixel samples from.

        Values outside ``[0, size - 1]`` contribute nothing (the tap is dropped) rather
        than being folded back onto another row, which is what the old check of the
        *linear* index got wrong.
        """
        return self.source_pixels.copy()

    # ----------------------------------------------------------------- applying
    def __call__(self, image):
        """PIL image in, PIL image out (also accepts/returns numpy arrays)."""
        from PIL import Image

        as_array = isinstance(image, np.ndarray)
        array = image if as_array else np.array(image.convert("RGB"))
        out = self.applyToArray(array)
        return out if as_array else Image.fromarray(out, "RGB")

    def applyToArray(self, array):
        """Resample a ``H x W x 3`` uint8/float array with this effector's map."""
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                "FisheyeEffector expects an H x W x 3 image, got shape {}".format(array.shape)
            )
        if array.shape[:2] != (self.height, self.width):
            # dtype preserving, and no PIL round trip (which only accepts float images in
            # a narrow range); nearest is fine because callers pass matching sizes
            array = nearestResize(array, self.height, self.width)

        org_dtype = array.dtype
        pixels = torch.from_numpy(np.ascontiguousarray(array).reshape(-1, 3)).float()
        sampled = pixels[self.tapIndex]                      # (4, H*W, 3)
        out = (sampled * self.tapWeight).sum(dim=0)
        out = out.reshape(self.height, self.width, 3)
        if org_dtype == np.uint8:
            return out.round().clamp(0, 255).to(torch.uint8).numpy()
        return out.numpy().astype(org_dtype)

    def apply(self, image_bytes):
        """Legacy helper: encoded image bytes in, PNG bytes out."""
        import io

        from PIL import Image

        return self.calcImage(Image.open(io.BytesIO(image_bytes)))

    def calcImage(self, image):
        """Legacy helper: a PIL image (or numpy array) in, PNG bytes out."""
        import io

        from PIL import Image

        out = self(image)
        if not isinstance(out, Image.Image):
            out = Image.fromarray(out, "RGB")
        buffer = io.BytesIO()
        out.save(buffer, format="PNG")
        return buffer.getvalue()


def nearestResize(array, height, width):
    """Fit an image to the effector frame (only needed when the sizes differ).

    uint8 goes through PIL with Lanczos, as before; other dtypes (float images, e.g.
    already-normalised tensors) use a dtype preserving nearest resample, because PIL
    will not build an RGB image out of float32.
    """
    if array.dtype == np.uint8:
        from PIL import Image

        image = Image.fromarray(np.ascontiguousarray(array), "RGB")
        return np.array(image.resize((width, height), Image.LANCZOS))

    rows = np.clip((np.arange(height) * array.shape[0] / height).astype(int),
                   0, array.shape[0] - 1)
    cols = np.clip((np.arange(width) * array.shape[1] / width).astype(int),
                   0, array.shape[1] - 1)
    return array[np.ix_(rows, cols)]


# ------------------------------------------------------------------ legacy helpers
def calc_points_of_original_image(x, y, r, distortion):
    """Deprecated shim for the old module level API (rectilinear source of a pixel)."""
    del r  # the division model needs only the coordinate itself
    out = distorted_to_original(
        torch.tensor([x, y], dtype=torch.float64),
        clamp_distortion(torch.tensor([distortion], dtype=torch.float64)),
    )
    return float(out[0]), float(out[1])


def calc_points_of_distorted_image(x, y, r, distortion):
    """Deprecated shim for the old module level API (where original content lands)."""
    del r
    out = original_to_distorted(
        torch.tensor([x, y], dtype=torch.float64),
        clamp_distortion(torch.tensor([distortion], dtype=torch.float64)),
    )
    return float(out[0]), float(out[1])


def outputSuffix(effector):
    """``_rec`` for rectification, ``_dis`` for rendering -- also what we skip on reruns."""
    return "_rec.png" if getattr(effector, "backward", False) else "_dis.png"


def execDir(effector, path, recursive=True):
    import os

    suffix = outputSuffix(effector)
    for file in sorted(os.listdir(path)):
        joined_path = os.path.join(path, file)
        if os.path.isfile(joined_path):
            if joined_path.endswith(suffix):
                continue
            base, _ = os.path.splitext(joined_path)
            execFile(effector, joined_path, output_path=base + suffix)
        elif recursive:
            execDir(effector, joined_path)


def execFile(effector, input_path, output_path="output.png"):
    import os

    if input_path.endswith(outputSuffix(effector)) or os.path.exists(output_path):
        print("skip", input_path)
        return

    with open(input_path, "rb") as image_bin:
        print(input_path)
        with open(output_path, "wb") as output:
            output.write(effector.apply(image_bin.read()))


def resizeAndApply(effector, width, height, input_path, output_path="output.png"):
    from PIL import Image

    print(input_path)
    image = Image.open(input_path).convert("RGB").resize((width, height))
    result = effector(image)
    if not isinstance(result, Image.Image):
        result = Image.fromarray(result, "RGB")
    result.save(output_path)


if __name__ == "__main__":
    import argparse
    import os
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to the input image or directory")
    parser.add_argument(
        "-d", "--distortion", type=float, default=0.1,
        help="amount of distortion between -1 and 1 (0.1 as default)",
    )
    parser.add_argument("--width", type=int, default=1280, help="input image width (1280)")
    parser.add_argument("--height", type=int, default=720, help="input image height (720)")
    parser.add_argument(
        "--backward", action="store_true",
        help="rectify the input (fisheye -> rectilinear) instead of distorting it",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("No such file or directory: {}".format(args.input))
        raise SystemExit(1)

    start = time.time()
    effector = FisheyeEffector(
        height=args.height, width=args.width, distortion=args.distortion,
        backward=args.backward, verbose=True,
    )
    print("{} sec had been spent for Initialize.".format(time.time() - start))

    import os.path as _p

    if os.path.isfile(args.input):
        base, _ = _p.splitext(args.input)
        out = base + outputSuffix(effector)
        resizeAndApply(effector, args.width, args.height, args.input, output_path=out)
    else:
        execDir(effector, args.input)
