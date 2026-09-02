"""Single-parameter division-model geometry, shared by the renderer and the loss.

Keeping the maths in one place is the point of this module: the effector, the key
coordinate labels and ``DistortionLoss`` must use *exactly* the same mapping, or the
network is trained towards a value that is not the distortion that generated the
image (which is what the previous version of ``core/losses.py`` did -- it re-applied
the forward map instead of inverting it).

Conventions
-----------
Coordinates are normalised **isotropic** units of a ``W x H`` image::

    x = (2 * col - W) / W                 -> (-1, 1)
    y = (2 * row - H) / H * (H / W)       -> (-H/W, H/W)

i.e. one unit is half the image width on *both* axes, so the isotropic radius is
simply ``sqrt(x**2 + y**2)``.

Models
------
forward (a.k.a. "how a fisheye image was rendered"), from a distorted coordinate::

    original = distorted / (1 - k * |distorted|**2)                -- distorted_to_original

its exact closed form inverse, from an original coordinate::

    distorted = original * 2 / (1 + sqrt(1 + 4 * k * |original|**2))   -- original_to_distorted

The second one is the rationalised form of ``(-1 + sqrt(1 + 4*k*U**2)) / (2*k*U)``.
It is used instead of that textbook quadratic formula on purpose: the latter is
``0/0`` at ``k = 0`` and loses every bit of precision for small ``|k|`` (in float32 it
returns garbage as soon as ``|k| <~ 1e-4``, which is exactly the regime the first
curriculum stage trains in).
"""

import torch

#: |k| is clamped to this value everywhere, matching the effector's historical limit.
MAX_DISTORTION = 1.0

#: Lower bound on ``1 - k * r**2``; the division model is singular at that surface.
MIN_DENOMINATOR = 1e-2


def clamp_distortion(distortion, limit=MAX_DISTORTION):
    """Constrain the single model parameter to a range the mapping can represent."""
    return torch.clamp(distortion, -limit, limit)


def _broadcast(distortion, coords):
    """``(B,)`` parameter against ``(..., N, 2)`` coordinates -> ``(B, 1, ...)``.

    Reshaping (rather than ``unsqueeze(-1)``) keeps one parameter usable for both the
    per-pixel grids of the renderer, ``(H*W, 2)``, and batched labels, ``(B, 3, 2)``.
    """
    flat = distortion.reshape(-1)
    if flat.numel() == 1:
        return flat.view((1,) * coords.dim())
    return flat.view((-1,) + (1,) * (coords.dim() - 1))


def _k_radius_sq(coords, distortion):
    """``distortion * |coords|^2``, broadcast over the leading dimensions of ``coords``."""
    return coords.pow(2).sum(dim=-1, keepdim=True) * _broadcast(distortion, coords)


def distorted_to_original(coords, distortion, min_denominator=MIN_DENOMINATOR):
    """Apply the division model: fisheye coordinate -> rectilinear coordinate.

    This is the map a fisheye image is *sampled* with when it is rendered, so it is
    also the map that rectifies one, and the one the training loss must use.
    ``distortion == 0`` is the identity and is numerically exact.
    """
    denom = 1.0 - _k_radius_sq(coords, distortion)
    # keep the sign (a negative denominator means the fold-over region, which is
    # reported as such rather than as inf) and bound the magnitude away from the
    # singular surface
    bounded = denom.abs().clamp(min=min_denominator)
    return coords / torch.where(denom < 0, -bounded, bounded)


def original_to_distorted(coords, distortion, max_discriminant=0.0):
    """Exact inverse of :func:`distorted_to_original`: rectilinear -> fisheye.

    Used to render a fisheye image (where does rectilinear content land?) and to
    project the key coordinates into the distorted image when labels are built.
    """
    disc = (1.0 + 4.0 * _k_radius_sq(coords, distortion)).clamp(min=max_discriminant)
    return coords * (2.0 / (1.0 + torch.sqrt(disc)))


def valid_radius(distortion):
    """Largest isotropic radius whose rectilinear source is still inside ``|o| <= 1``.

    ``|o| = r / (1 - k r^2)``, so for ``k > 0`` the bound solves ``k r^2 + r - 1 = 0``.
    Returns ``None`` when the whole frame is valid (``k <= 0``: pincushion pulls
    content inwards, it never samples outside the input).
    """
    if distortion <= 0:
        return None
    k = min(float(distortion), MAX_DISTORTION)
    return (-1.0 + (1.0 + 4.0 * k) ** 0.5) / (2.0 * k)
