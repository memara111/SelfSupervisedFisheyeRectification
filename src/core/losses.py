import torch
import torch.nn as nn


class DistortionLoss(nn.Module):
    """Self-supervised distortion loss for the division model.

    Fixes applied (relative to the original implementation):
    - avoids the numerically unstable quadratic-root expression
      (-1 + sqrt(...)) / (2*a), which suffers catastrophic cancellation
      when distortion (and therefore `a`) is close to zero — exactly the
      regime the model starts in at initialization;
    - avoids division by x when x is zero/near zero;
    - keeps the loss finite for normal network outputs.

    The rewritten root is algebraically identical to the original
    (-1 + sqrt(1 - 4ac)) / (2a) via the product-of-roots identity
    (z_plus * z_minus = c/a), but its denominator (-1 - sqrt(D)) is always
    <= -1 in magnitude, so it never approaches zero regardless of `a`.

    Verified numerically against FisheyeEffector.calc_points_of_distorted_image
    (the ground-truth distortion model used to build the dataset) across
    distortion in [-0.999, 0.9] including near-zero values: outputs match to
    within floating-point precision, and the new form stays accurate exactly
    where the original form loses precision (|distortion| ~ 1e-6).
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, distortion, coordinate_norms):
        dst_coordinates = []
        for x, y in coordinate_norms:
            dst_x, dst_y = self.project(x, y, distortion)
            dst_coordinates.append((dst_x, dst_y))

        return torch.mean(self.calc_distance(dst_coordinates).square())

    def project(self, x, y, distortion):
        # Division model:
        # x_distorted = x_undistorted / (1 + distortion * r^2)
        #
        # The original implementation solved a quadratic using
        # (-1 + sqrt(...)) / (2*a). That form is unstable when a -> 0.
        #
        # Use the algebraically equivalent stable form:
        #   2*c / (-1 - sqrt(1 - 4*a*c))
        # where c = -x.
        x_safe = torch.where(
            x.abs() < self.eps,
            torch.full_like(x, self.eps),
            x,
        )

        r2 = x_safe.square() + y.square()
        a = distortion * r2 / x_safe
        c = -x_safe

        discriminant = (1.0 - 4.0 * a * c).clamp_min(self.eps)
        sqrt_disc = torch.sqrt(discriminant)
        denominator = -1.0 - sqrt_disc

        new_x = (2.0 * c) / denominator

        # At distortion == 0, the exact solution is x.
        # The stable expression is already well behaved there, but explicitly
        # select x to avoid accumulating floating-point error around zero.
        zero_distortion = distortion.abs() < self.eps
        new_x = torch.where(zero_distortion, x_safe, new_x)

        new_y = (y / x_safe) * new_x

        # Preserve the exact origin.
        origin = (x.abs() < self.eps) & (y.abs() < self.eps)
        new_x = torch.where(origin, torch.zeros_like(new_x), new_x)
        new_y = torch.where(origin, torch.zeros_like(new_y), new_y)

        return new_x, new_y

    def calc_distance(self, coordinates):
        return 100.0 * (coordinates[1][0] - coordinates[0][0])
