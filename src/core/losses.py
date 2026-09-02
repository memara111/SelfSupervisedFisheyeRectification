import torch
import torch.nn as nn

from core.division_model import clamp_distortion, distorted_to_original


class DistortionLoss(nn.Module):
    """Self-supervision by *reconstructing coordinate relations*.

    The dataset hands over three points that are known to be vertically aligned and
    equally spaced in the rectilinear image (the effector projected them into the
    fisheye image, which is the only supervision available). The network has to output
    the single parameter that, when the *rectification* map is applied to those
    points, restores that relation.

    Two things the previous implementation got wrong and that this class fixes:

    * it applied ``original_to_distorted`` -- the *same* map the labels were built
      with -- so it minimised a double distortion. The optimum then sat at a
      parameter of the wrong sign (measured: ``k = -0.295`` for a ground truth of
      ``k = +0.9``). Rectifying with ``distorted_to_original`` puts the minimum exactly
      on the ground truth and is also well defined at ``k = 0``, where the quadratic
      formula used to produce ``0/0`` -> NaN loss -> NaN weights.
    * it only used ``x`` of the first two points. One constraint for one unknown has
      several roots, which is what made the objective ambiguous. Using both x
      constraints plus the y spacing makes the ground truth the only zero.

    The key points are given in the *delivered* frame, i.e. after the effector cropped
    and magnified the render; that magnification has to be undone before the model is
    applied, because the division model is not scale invariant.
    """

    def __init__(self, coordinate_scale=100.0, max_distortion=None, y_weight=1.0):
        """``max_distortion`` optionally clamps the parameter to the range the mapping can
        represent (``None`` keeps every prediction in play; the division model itself is
        protected against its singular surface in ``core.division_model``)."""
        super().__init__()
        self.scale = float(coordinate_scale)
        self.max_distortion = None if max_distortion is None else float(max_distortion)
        self.y_weight = float(y_weight)

    @staticmethod
    def _unpack(labels):
        """Accept the collated batch as a dict, an (coords, expansion) pair or bare coords."""
        coords, expansion = labels, None
        if isinstance(labels, dict):
            coords = labels["coords"]
            expansion = labels.get("expansion")
        elif isinstance(labels, (tuple, list)) and len(labels) == 2:
            coords, expansion = labels
            if not (torch.is_tensor(coords) and torch.is_tensor(expansion)):
                raise ValueError(
                    "DistortionLoss expected (coords, expansion) tensors, got {}".format(
                        [type(x).__name__ for x in labels]
                    )
                )
        if not torch.is_tensor(coords):
            raise ValueError("DistortionLoss expects key coordinate tensors, got {}".format(
                type(coords).__name__))
        if coords.dim() != 3 or tuple(coords.shape[-2:]) != (3, 2):
            raise ValueError(
                "DistortionLoss expects key coordinates of shape (B, 3, 2), got {}".format(
                    tuple(coords.shape)
                )
            )
        if expansion is not None:
            expansion = expansion.reshape(-1)
            if expansion.numel() != coords.shape[0]:
                raise ValueError(
                    "expansion must hold one value per sample: got {} for batch {}".format(
                        expansion.numel(), coords.shape[0]
                    )
                )
        return coords, expansion

    def forward(self, distortion, labels):
        coords, expansion = self._unpack(labels)

        k = distortion.reshape(-1)
        if self.max_distortion is not None:
            k = clamp_distortion(k, self.max_distortion)
        if k.shape[0] == 1 and coords.shape[0] != 1:
            k = k.expand(coords.shape[0])
        if k.shape[0] != coords.shape[0]:
            raise ValueError(
                "got {} predictions for {} samples".format(k.shape[0], coords.shape[0])
            )

        points = coords.to(dtype=k.dtype)  # (B, 3, 2) in the delivered frame
        if expansion is not None:
            points = points / expansion.to(dtype=k.dtype).view(-1, 1, 1)

        rectified = distorted_to_original(points, k)  # (B, 3, 2)

        x = rectified[..., 0]
        y = rectified[..., 1]
        align = (x[:, 1] - x[:, 0]).pow(2) + (x[:, 2] - x[:, 1]).pow(2)
        spacing = (y[:, 0] - y[:, 1]).sub(y[:, 1] - y[:, 2]).pow(2)
        residual = align + self.y_weight * spacing

        return (self.scale ** 2) * torch.mean(residual)
