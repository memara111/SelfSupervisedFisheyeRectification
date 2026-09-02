"""Regression tests for the fixes in this branch. Runnable two ways:

    python3 tests/run_tests.py        # no extra dependency
    python3 -m pytest tests/run_tests.py

Everything here works on synthetic images -- no dataset required. Tests that need
torchvision (the VGG head) are skipped when it is missing, so the file can be used as a
smoke check in a bare checkout.
"""

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (os.path.join(REPO, "src"), os.path.join(REPO, "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

from PIL import Image  # noqa: E402

from core.config import Config  # noqa: E402
from core.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from core.functions import getDistortions, curriculumDistortions, val  # noqa: E402
from core.losses import DistortionLoss  # noqa: E402
from core.division_model import distorted_to_original, original_to_distorted  # noqa: E402
from datasets import FisheyeEffector, DistortDataset, BaseDataset, EvalDataset  # noqa: E402

H, W = 64, 128


def stripe(col=86):
    """One sharp vertical bar: good for measuring where a straight line ends up."""
    a = np.zeros((H, W, 3), np.uint8)
    a[:, col - 2:col + 2] = 255
    return a


def barStats(arr):
    """(mean column of the bar over all rows, spread of that column -> 0 if straight)"""
    g = np.asarray(arr).astype(float).mean(axis=2)
    cols = []
    for row in range(g.shape[0]):
        hits = np.where(g[row] > 128)[0]
        if hits.size:
            cols.append(hits.mean())
    if not cols:
        return float("nan"), float("nan")
    return float(np.mean(cols)), float(np.ptp(cols))


def smooth():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    pattern = 128 + 50 * np.sin(xx / 9.0) * np.cos(yy / 7.0)
    return np.repeat(pattern[..., None], 3, axis=2).clip(0, 255).astype(np.uint8)


def bowing(arr):
    return barStats(arr)[1]


def toTensor(image):
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr.transpose(2, 0, 1))


def have(module):
    return importlib.util.find_spec(module) is not None


# --------------------------------------------------------------------------- config
def test_config_uses_safe_load_and_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg.yml")
        with open(path, "w") as f:
            f.write("DATASET:\n  NAME: x\n  HEIGHT: 32\n")
        config = Config(path, defaults={"DATASET": {"WIDTH": 512}, "TRAIN": {"BATCH_SIZE": 4}})
        assert config.get("DATASET.NAME") == "x"
        assert config.get("DATASET.WIDTH") == 512, "defaults must fill absent keys"
        assert config.get("TRAIN.CURRICULUM.MAX_DISTORTION", 0.9) == 0.9
        assert config["DATASET"]["HEIGHT"] == 32
        try:
            Config(os.path.join(tmp, "nope.yml"))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing config file must raise, not exit(1)")


def test_shipped_config_is_loadable():
    config = Config(os.path.join(REPO, "cfg", "cityscapes.yml"))
    assert config.get("DATASET.NAME") == "cityscapes"
    assert config.get("TRAIN.CURRICULUM.SWITCH_EPOCH") == 30
    assert config.get("DATASET.CROP") is True
    # the data range must stay strictly inside the model bound, or the hardest targets
    # sit where the squashed head cannot reach them
    bound = config.get("MODEL.MAX_DISTORTION")
    assert bound == 1.0, bound
    assert config.get("TRAIN.CURRICULUM.MAX_DISTORTION") < bound
    assert config.get("TEST.MAX_DISTORTION") < bound


# ---------------------------------------------------------------------------- model
def test_division_model_is_an_exact_involution():
    pts = torch.tensor([[0.0, 0.0], [0.2, 0.1], [0.2, -0.1], [0.9, 0.4], [-0.7, 0.7]],
                       dtype=torch.float64)
    for k in [-0.9, -0.5, -1e-4, 0.0, 1e-4, 0.05, 0.5, 0.9]:
        kk = torch.tensor([k], dtype=torch.float64)
        back = original_to_distorted(distorted_to_original(pts, kk), kk)
        assert torch.allclose(pts, back, atol=1e-12), (k, pts, back)


def test_division_model_is_stable_in_float32():
    # (-1 + sqrt(1 + tiny)) / (2 * tiny) is the form that used to blow up here
    for k in [0.0, 1e-6, 1e-4, 1e-2]:
        kk = torch.tensor([k], dtype=torch.float32)
        pts = torch.tensor([[0.2, 0.1]], dtype=torch.float32)
        rectified = distorted_to_original(original_to_distorted(pts, kk), kk)
        assert torch.isfinite(rectified).all()
        assert (rectified - pts).abs().max().item() < 1e-6, k


def test_valid_radius_bounds_the_crop():
    from core.division_model import valid_radius

    assert valid_radius(0.0) is None and valid_radius(-0.5) is None
    assert 0.0 < valid_radius(0.9) < 1.0
    # a rectilinear point at the valid radius maps to the frame edge
    r = valid_radius(0.4)
    edge = distorted_to_original(torch.tensor([[r, 0.0]], dtype=torch.float64),
                                torch.tensor([0.4], dtype=torch.float64))
    assert abs(float(edge[0, 0]) - 1.0) < 1e-9, edge


def test_get_distortions_covers_both_signs():
    assert getDistortions(3) == [0.0, 0.45, 0.9]
    symmetric = getDistortions(5, symmetric=True, max_distortion=0.6)
    assert symmetric[0] == -0.6 and symmetric[-1] == 0.6
    # the deterministic schedule used to be pinned to [0, 0.9] whatever the fragment
    # count; a curriculum has to grow
    first = curriculumDistortions(0, 30, max_distortion=0.9, start_distortion=0.3)
    later = curriculumDistortions(29 * 30, 30, max_distortion=0.9, start_distortion=0.3)
    assert max(abs(d) for d in first) < max(abs(d) for d in later)
    assert len(first) < len(later)


# ----------------------------------------------------------------------------- loss
def _labels(gt_k, height=H, width=W):
    eff = FisheyeEffector(height=height, width=width, distortion=gt_k)
    coords = torch.tensor(eff.getKeyCoordinates(), dtype=torch.float64)[None]
    expansion = torch.tensor([eff.getExpansion()], dtype=torch.float64)
    return (coords, expansion)


def test_loss_is_minimised_at_the_ground_truth():
    torch.set_default_dtype(torch.float64)
    try:
        criterion = DistortionLoss()
        for gt in [0.0, 0.05, 0.2, 0.45, 0.9, -0.3, -0.9]:
            labels = _labels(gt)
            grid = torch.linspace(-1, 1, 4001, dtype=torch.float64)
            values = torch.stack([criterion(k.reshape(1), labels) for k in grid])
            assert torch.isfinite(values).all(), gt
            argmin = float(grid[int(torch.argmin(values))])
            assert abs(argmin - gt) < 5e-4, (gt, argmin)
            assert float(criterion(torch.tensor([gt]), labels)) < 1e-9, gt
    finally:
        torch.set_default_dtype(torch.float32)


def test_loss_is_finite_and_small_at_zero_distortion():
    """The k = 0 fragment of the curriculum used to give loss = nan -> nan weights."""
    for dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
        try:
            labels = _labels(0.0)
            criterion = DistortionLoss()
            for k in [0.0, 1e-6, 1e-3, 1e-2]:
                pred = torch.tensor([k], dtype=dtype, requires_grad=True)
                loss = criterion(pred, labels)
                assert torch.isfinite(loss), (dtype, k, loss)
                loss.backward()
                assert torch.isfinite(pred.grad).all(), (dtype, k, pred.grad)
                assert float(loss) < 1e-3, (dtype, k, float(loss))
            # approaching the truth must decrease the loss, not explode near it
            losses = [float(criterion(torch.tensor([k], dtype=dtype), labels))
                      for k in (0.2, 0.05, 0.01, 0.0)]
            assert losses == sorted(losses, reverse=True), (dtype, losses)
        finally:
            torch.set_default_dtype(torch.float32)


def test_loss_accepts_dict_and_tuple_labels():
    coords, expansion = _labels(0.4)
    criterion = DistortionLoss()
    as_dict = {"coords": coords, "expansion": expansion}
    pred = torch.tensor([0.4], dtype=torch.float64)
    assert torch.allclose(criterion(pred, as_dict), criterion(pred, (coords, expansion)))
    bare = coords / expansion.view(-1, 1, 1)
    assert torch.allclose(criterion(pred, bare), criterion(pred, (coords, expansion)),
                          atol=1e-9)
    try:
        criterion(pred, torch.zeros(1, 4, 2, dtype=torch.float64))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong label shape should raise")


def test_loss_is_correct_for_real_batches():
    """Batch > 1 used to be a broadcasting error (only ever exercised with batch 1)."""
    torch.set_default_dtype(torch.float64)
    try:
        criterion = DistortionLoss()
        gts, coords, exps = [0.0, 0.35, -0.6, 0.9], [], []
        for gt in gts:
            c, e = _labels(gt)
            coords.append(c[0])
            exps.append(e.reshape(1))
        coords = torch.stack(coords)                      # (4, 3, 2)
        exps = torch.cat(exps)                            # (4,)
        wrong = torch.tensor([g + 0.3 for g in gts], dtype=torch.float64)

        batch = criterion(wrong, (coords, exps))
        singles = torch.stack([criterion(wrong[i:i + 1], (coords[i:i + 1], exps[i:i + 1]))
                              for i in range(len(gts))])
        assert torch.allclose(batch, singles.mean(), atol=1e-12), (batch, singles)

        # each sample must be judged only by its own geometry, and be closest to it
        right = criterion(torch.tensor(gts, dtype=torch.float64), (coords, exps))
        assert float(right) < 1e-9, float(right)
        for i, gt in enumerate(gts):
            per_sample = float(criterion(torch.tensor([gt], dtype=torch.float64),
                                        (coords[i:i + 1], exps[i:i + 1])))
            assert per_sample < 1e-9, (i, gt, per_sample)
        assert float(batch) > 1e3 * float(right), (float(batch), float(right))
    finally:
        torch.set_default_dtype(torch.float32)


def test_loss_is_optimisable_to_the_ground_truth():
    """The point of the rewrite: plain Adam on the objective recovers k.

    With the old forward-map loss the same optimisation converged to a parameter of the
    *wrong sign* (k = -0.295 for a ground truth of +0.9) or stalled at |k| ~ 9e-3
    because of the 0/0 cancellation at k = 0.
    """
    criterion = DistortionLoss()
    for gt in [0.0, 0.2, 0.5, -0.4, 0.9]:
        coords, expansion = _labels(gt)
        coords = coords.float()
        expansion = expansion.float()
        param = torch.tensor([0.9 if gt <= 0 else -0.9], requires_grad=True)  # start wrong
        optimizer = torch.optim.Adam([param], lr=1e-2)
        for _ in range(3000):
            optimizer.zero_grad()
            loss = criterion(param, (coords, expansion))
            assert torch.isfinite(loss), (gt, loss)
            loss.backward()
            optimizer.step()
        assert abs(float(param) - gt) < 0.03, (gt, float(param), float(loss))


def test_train_skips_non_finite_batches_without_stepping():
    """A NaN batch must not reach optimizer.step(), or the weights never recover."""
    from core.functions import train as train_step

    model = torch.nn.Linear(2, 1)
    model.weight.data.fill_(0.25)
    model.bias.data.zero_()
    before = model.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = (torch.ones(2, 2), {"coords": torch.zeros(2, 3, 2)})

    class NaN:
        def __call__(self, outputs, labels):
            return outputs.sum() * float("nan")

    loss = train_step(model=model, dataloader=[batch, batch], criterion=NaN(),
                      optimizer=optimizer, device="cpu", log_interval=0)
    assert math.isnan(loss), loss
    assert torch.equal(model.weight.detach(), before), "NaN batch moved the weights"
    assert torch.isfinite(model.weight).all()

    finite = train_step(model=model, dataloader=[batch, batch],
                        criterion=lambda o, l: o.sum() ** 2, optimizer=optimizer,
                        device="cpu", log_interval=0)
    assert math.isfinite(finite) and finite >= 0.0, finite
    assert not torch.equal(model.weight.detach(), before), "a finite batch must step"


def test_val_survives_an_empty_loader():
    class Empty:
        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0

    assert math.isnan(val(model=torch.nn.Identity(), dataloader=Empty(),
                         criterion=torch.nn.Identity()))


# -------------------------------------------------------------------------- effector
def test_effector_bounds_are_per_axis_not_wrapped():
    uncropped = FisheyeEffector(height=H, width=W, distortion=0.7, crop=False)
    src = uncropped.sourcePixels()  # (H*W, 2) float pixel coordinates of every sample
    out_of_frame = (src[:, 0] < 0.0) | (src[:, 0] > W - 1.0) | \
                   (src[:, 1] < 0.0) | (src[:, 1] > H - 1.0)
    assert out_of_frame.any(), "an uncropped 0.7 render must have pixels with no source"

    # row 0 content must never resurface in the lower half (that is what the old
    # linear-index check did: a negative row index aliased onto a valid pixel)
    bright = np.zeros((H, W, 3), np.uint8)
    bright[0, :] = 255
    rendered = uncropped(bright)
    assert rendered[H // 2:, :].max() == 0, "row 0 content aliased into the lower half"
    assert float(rendered.max()) <= 255.0

    for eff in (uncropped, FisheyeEffector(height=H, width=W, distortion=0.7)):
        sums = eff.tapWeight.squeeze(-1).sum(dim=0)      # total weight per output pixel
        assert float(sums.max()) <= 1.0 + 1e-5, float(sums.max())
        assert float(sums.min()) >= 0.0
        # convex weights: a pixel with an in-frame source keeps all of its energy, a
        # pixel that falls off the edge is attenuated (never wrapped around)
        src = eff.sourcePixels()
        partial = ((src[:, 0] < 0) | (src[:, 0] > W - 1) |
                   (src[:, 1] < 0) | (src[:, 1] > H - 1))
        assert int((sums < 1.0 - 1e-6).sum()) == int(partial.sum()), (
            int((sums < 1.0 - 1e-6).sum()), int(partial.sum()))
        assert eff.tapIndex.shape == (4, H * W) and eff.tapWeight.shape == (4, H * W, 1)
    # the crop guarantees every delivered pixel has an in-frame source
    cropped = FisheyeEffector(height=H, width=W, distortion=0.7).sourcePixels()
    assert not ((cropped[:, 0] < 0.0) | (cropped[:, 0] > W - 1.0) |
                (cropped[:, 1] < 0.0) | (cropped[:, 1] > H - 1.0)).any()


def test_crop_is_aspect_preserving():
    """The crop must be proportional to the frame, else one scale factor stretches it.

    Two checks: the sampled region's corner sits exactly on the valid circle, and a
    square patch survives render + rectify as a square.
    """
    from core.division_model import valid_radius

    for height, width in ((64, 128), (128, 64), (100, 100), (256, 512)):
        for k in (0.2, 0.9):
            eff = FisheyeEffector(height=height, width=width, distortion=k)
            aspect = height / width
            corner = math.hypot(1.0 / eff.expansion, aspect / eff.expansion)
            assert abs(corner - valid_radius(k)) < 1e-12, (height, width, k, corner)

            a = np.zeros((height, width, 3), np.uint8)
            cy, cx = height // 2, width // 2
            a[cy - 10:cy + 10, cx - 10:cx + 10] = 255
            back = FisheyeEffector(height=height, width=width, distortion=k,
                                  backward=True)(eff(a.copy()))
            hits = np.asarray(back).mean(axis=2) > 128
            rows, cols = np.where(hits)
            got = (cols.max() - cols.min() + 1) / float(rows.max() - rows.min() + 1)
            assert abs(got - 1.0) < 0.12, (height, width, k, got)


def test_render_then_rectify_recovers_the_image():
    image = smooth()
    for k in [0.05, 0.3, 0.7]:
        forward = FisheyeEffector(height=H, width=W, distortion=k)
        fisheye = forward(image.copy())
        assert fisheye.max() > 0 and not np.array_equal(fisheye, image), k
        rectified = FisheyeEffector(height=H, width=W, distortion=k, backward=True)(fisheye)
        inner = (slice(H // 4, 3 * H // 4), slice(W // 4, 3 * W // 4))
        diff = float(np.abs(rectified[inner].astype(int) - image[inner].astype(int)).mean())
        assert diff < 8, (k, diff)


def test_rectify_is_the_inverse_not_a_second_distortion():
    """test.py used to apply the render map again, doubling the distortion."""
    k = 0.7
    original = stripe()
    original_pos = barStats(original)[0]


    fisheye = FisheyeEffector(height=H, width=W, distortion=k)(original)
    # the render has to move and bend the bar, otherwise the test proves nothing
    assert abs(barStats(fisheye)[0] - original_pos) > 5.0, barStats(fisheye)
    assert bowing(fisheye) > 1.0, barStats(fisheye)

    fixed = FisheyeEffector(height=H, width=W, distortion=k, backward=True)(fisheye)
    doubled = FisheyeEffector(height=H, width=W, distortion=k)(fisheye)
    assert abs(barStats(fixed)[0] - original_pos) < 2.0, (barStats(fixed), original_pos)
    assert bowing(fixed) < 1.0, barStats(fixed)
    # the double warp pushes the bar out of the frame entirely
    mse = lambda a: float(((np.asarray(a).astype(int) - original.astype(int)) ** 2).mean())
    assert mse(doubled) > 4.0 * mse(fixed), (mse(doubled), mse(fixed))


def test_effector_accepts_other_sizes_and_dtypes():
    image = stripe()
    eff = FisheyeEffector(height=H, width=W, distortion=0.2)
    big = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)  # 2x the frame
    assert eff(big).shape == (H, W, 3)
    floats = image.astype(np.float32) / 255.0
    out = eff(floats)
    assert out.dtype == floats.dtype and out.shape == (H, W, 3)
    assert float(out.max()) <= 1.0 + 1e-6
    try:
        eff(np.zeros((H, W), np.uint8))
    except ValueError as error:
        assert "H x W x 3" in str(error)
    else:
        raise AssertionError("a grayscale array must be rejected with a clear message")


def test_identity_at_zero_distortion():
    image = stripe()
    eff = FisheyeEffector(height=H, width=W, distortion=0.0)
    assert np.array_equal(eff(image.copy()), image), "k=0 must be the identity"
    assert np.array_equal(
        FisheyeEffector(height=H, width=W, distortion=0.0, backward=True)(image.copy()), image
    )


# -------------------------------------------------------------------------- datasets
def _writeImages(tmp, names=("a_rgb.png", "b_rgba.png", "c_gray.png")):
    paths = []
    for i, name in enumerate(names):
        arr = np.clip(stripe().astype(int) + i * 4, 0, 255).astype(np.uint8)
        path = os.path.join(tmp, name)
        image = Image.fromarray(arr, "RGB")
        if i == 1:
            image = image.convert("RGBA")  # alpha on purpose
        if i == 2:
            image = image.convert("L")  # grayscale on purpose
        image.save(path)
        paths.append(path)
    return paths


def test_distort_dataset_handles_any_input_mode():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _writeImages(tmp)
        lst = os.path.join(tmp, "train.lst")
        with open(lst, "w") as f:
            f.write("\n".join(paths) + "\n\n")  # trailing blank line on purpose
        out = os.path.join(tmp, "out")
        dataset = DistortDataset(list_path=lst, height=H, width=W, transform=toTensor,
                                 distortions=[0.2, -0.4], output_dir=out)
        assert len(dataset) == 3
        image, label = dataset[0]
        assert tuple(image.shape) == (3, H, W), image.shape
        assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
        assert tuple(label["coords"].shape) == (3, 2)
        assert label["expansion"].dim() == 0
        assert round(float(label["distortion"]), 6) in (0.2, -0.4), label["distortion"]
        assert os.path.isdir(out) and len(os.listdir(out)) == 2
        Image.open(os.path.join(out, sorted(os.listdir(out))[0])).convert("RGB")


def test_dataset_is_deterministic_when_asked():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _writeImages(tmp, names=("a.png",))
        lst = os.path.join(tmp, "l.lst")
        with open(lst, "w") as f:
            f.write(paths[0] + "\n")
        kwargs = dict(list_path=lst, height=H, width=W, transform=toTensor,
                      distortions=[0.1, 0.2, 0.3, 0.4])
        det = DistortDataset(deterministic=True, **kwargs)
        got = {round(float(det[0][1]["distortion"]), 6) for _ in range(12)}
        assert got == {0.1}, got
        rand = DistortDataset(deterministic=False, **kwargs)
        seen = {round(float(rand[0][1]["distortion"]), 6) for _ in range(60)}
        assert len(seen) > 1, seen


def test_base_dataset_returns_the_label_not_the_filename():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _writeImages(tmp, names=("a.png",))
        label_file = os.path.join(tmp, "7_label.txt")
        with open(label_file, "w") as f:
            f.write("7\n")
        lst = os.path.join(tmp, "l.lst")
        with open(lst, "w") as f:
            f.write("{} {}\n".format(paths[0], label_file))
        image, label = BaseDataset(list_path=lst)[0]
        assert label == 7, label
        assert image.size == (W, H)


def test_eval_dataset_filters_images_only():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _writeImages(tmp)
        with open(os.path.join(tmp, "notes.txt"), "w") as f:
            f.write("ignore me")
        os.makedirs(os.path.join(tmp, "rectified"))
        dataset = EvalDataset(data_path=tmp, transform=toTensor, height=H, width=W)
        assert len(dataset) == len(paths), dataset.img_list
        image, meta = dataset[0]
        assert tuple(image.shape) == (3, H, W)
        assert os.path.basename(meta["path"]).endswith((".png", ".jpg"))


# ------------------------------------------------------------------------------ net
def test_pem_output_is_bounded_and_backpropagates():
    if not have("torchvision"):
        return
    from models import ParametersEstimationModule

    model = ParametersEstimationModule(in_channels=3, vgg_pretrained=False)
    model.eval()
    out = model(torch.zeros(2, 3, H, W))
    assert tuple(out.shape) == (2,)
    assert out.abs().max() <= 1.0 + 1e-6, out
    out.sum().backward()
    grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_pem_freeze_vgg_leaves_only_the_head_trainable():
    if not have("torchvision"):
        return
    from models import ParametersEstimationModule

    model = ParametersEstimationModule(freeze_vgg=True)
    assert all(not p.requires_grad for p in model.vgg.parameters())
    assert any(p.requires_grad for p in model.encoder.parameters())
    model.train()
    assert not model.vgg.training, "frozen branch must stay in eval mode"


def test_checkpoint_loads_across_dataparallel_wrapping():
    src = torch.nn.Linear(4, 2)
    src.weight.data.fill_(3.5)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_checkpoint(os.path.join(tmp, "ck.pt"), torch.nn.DataParallel(src))
        dst = torch.nn.Linear(4, 2)
        load_checkpoint(path, dst, strict=True)
        assert torch.allclose(dst.weight, src.weight), "silent partial load (strict=False bug)"
        wrapped = torch.nn.DataParallel(torch.nn.Linear(4, 2))
        load_checkpoint(path, wrapped, strict=True)
        assert torch.allclose(wrapped.module.weight, src.weight)
        bogus = torch.nn.Linear(4, 3)
        try:
            load_checkpoint(path, bogus, strict=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a real mismatch must not pass silently")


# ---------------------------------------------------------------------- tools + e2e
def test_trained_range_is_recorded_and_enforced():
    from core.checkpoint import checkTrainedRange

    checkTrainedRange({"max_distortion": 0.85}, 0.85)      # legacy field name
    checkTrainedRange({"model_max_distortion": 1.0}, 1.0)
    checkTrainedRange({}, 0.9)                      # old checkpoints: no metadata
    checkTrainedRange(None, 0.9)
    try:
        checkTrainedRange({"model_max_distortion": 0.85}, 0.9)
    except RuntimeError as error:
        assert "MODEL.MAX_DISTORTION mismatch" in str(error)
    else:
        raise AssertionError("a rescaled evaluation must not pass silently")


def test_prepare_cityscapes_builds_the_list_files():
    import prepare_cityscapes as tool

    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "data", "cityscapes")
        for split, cities in (("train", ["aachen", "bochum"]), ("val", ["frankfurt"]),
                              ("test", ["berlin"])):
            for city in cities:
                d = os.path.join(root, "leftImg8bit_trainvaltest", "leftImg8bit", split, city)
                os.makedirs(d, exist_ok=True)
                for i in range(2):
                    with open(os.path.join(d, "{}_000{}.png".format(city, i)), "wb") as f:
                        f.write(b"x")
        assert tool.main(["--data_root", root, "--mode", "copy"]) == 0
        for split, expected in (("train", 4), ("val", 2), ("test", 2)):
            lst = os.path.join(root, "{}.lst".format(split))
            with open(lst) as f:
                lines = [line for line in f.read().split("\n") if line.strip()]
            assert len(lines) == expected, (split, lines)
            assert all(os.path.isabs(line) and os.path.isfile(line) for line in lines)
            assert os.path.isdir(os.path.join(root, split))
        with open(os.path.join(root, "train.lst")) as f:
            assert len([l for l in f.read().split("\n") if l.strip()]) == 4
        assert tool.main(["--data_root", root, "--mode", "copy"]) == 0, "must be idempotent"
        with open(os.path.join(root, "train.lst")) as f:
            assert len([l for l in f.read().split("\n") if l.strip()]) == 4
        dataset = DistortDataset(list_path=os.path.join(root, "val.lst"), height=H, width=W)
        assert len(dataset) == 2


def _fakeDataset(root, count=4, height=H, width=W):
    os.makedirs(root, exist_ok=True)
    paths = []
    for i in range(count):
        arr = np.zeros((height, width, 3), np.uint8)
        for x in range(10 + i, width, 25):
            arr[:, x:x + 2] = 220
        path = os.path.join(root, "img{}.png".format(i))
        Image.fromarray(arr, "RGB").save(path)
        paths.append(path)
    for split in ("train.lst", "val.lst", "test.lst"):
        with open(os.path.join(root, split), "w") as f:
            f.write("\n".join(paths) + "\n")
    return paths


def test_end_to_end_train_then_test_then_predict():
    """The whole documented flow. train.py used to die on `plt` *before* saving, so no
    checkpoint ever existed and test.py / predict.py could not run at all."""
    if not have("torchvision"):
        return
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"),
               MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES="")
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "data", "tiny")
        _fakeDataset(data)
        cfg = os.path.join(tmp, "tiny.yml")
        with open(cfg, "w") as f:
            f.write("DATASET:\n  NAME: tiny\n  HEIGHT: {}\n  WIDTH: {}\n"
                    "MODEL:\n  MAX_DISTORTION: 1.0\n"
                    "TRAIN:\n  MAX_EPOCH: 2\n  BATCH_SIZE: 2\n  LEARNING_RATE: 0.01\n"
                    "  NUM_WORKERS: 0\n  CURRICULUM:\n    ENABLED: true\n    SWITCH_EPOCH: 1\n"
                    "    MAX_FRAGMENTS: 3\nTEST:\n  CHECKPOINT: checkpoint.pth.tar\n"
                    "  SAVE_RESULTS: true\n  OUTPUT_DIR: {}\n  NUM_FRAGMENTS: 3\n"
                    "  OUTPUT_SIZE:\n    HEIGHT: {}\n    WIDTH: {}\n".format(
                        H, W, os.path.join(tmp, "results"), H, W))

        def run(script, *extra):
            proc = subprocess.run([sys.executable, os.path.join(REPO, "src", script), cfg]
                                  + list(extra), cwd=tmp, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True)
            assert proc.returncode == 0, "{} failed:\n{}".format(script, proc.stdout[-4000:])
            return proc.stdout

        out = run("train.py", "--data_root", os.path.join(tmp, "data"),
                  "--output_dir", os.path.join(tmp, "outputs", "tiny"))
        assert "Traceback" not in out, out[-3000:]
        assert os.path.isfile(os.path.join(tmp, "outputs", "tiny", "checkpoint.pth.tar"))
        with open(os.path.join(tmp, "outputs", "tiny", "losses.json")) as f:
            losses = json.load(f)
        # scan the pickle header instead of torch.load: a 129M-param checkpoint with
        # optimizer states is ~1.5 GB and would be held alongside the child process
        with open(os.path.join(tmp, "outputs", "tiny", "checkpoint.pth.tar"), "rb") as f:
            header = f.read(1 << 16)
        assert b"max_distortion" in header, "checkpoint must record the trained range"
        assert len(losses["train"]) == 2
        assert all(math.isfinite(v) for v in losses["train"] + losses["val"]), losses
        if have("matplotlib"):
            assert os.path.isfile(os.path.join(tmp, "outputs", "tiny", "losses.png"))

        out = run("test.py", "--data_root", os.path.join(tmp, "data"))
        assert "abs err" in out and "MAE" in out, out[-3000:]
        assert os.path.isfile(os.path.join(tmp, "results", "metrics.csv"))
        assert os.path.isfile(os.path.join(tmp, "results", "0_rec.jpg"))

        run("predict.py", "-i", data, "-ow", str(W), "-oh", str(H))
        assert len(os.listdir(os.path.join(data, "rectified"))) == 4

        out = run("train.py", "--data_root", os.path.join(tmp, "data"),
                  "--output_dir", os.path.join(tmp, "outputs", "tiny"), "--resume")
        assert "resuming from epoch 2" in out, out[-2000:]


def test_effector_cli_runs():
    """__main__ used to pass a `backward=` argument the constructor did not accept."""
    with tempfile.TemporaryDirectory() as tmp:
        Image.fromarray(stripe(), "RGB").save(os.path.join(tmp, "in.png"))
        for extra in ([], ["--backward"]):
            proc = subprocess.run(
                [sys.executable, os.path.join(REPO, "src", "datasets", "fisheye_effector.py"),
                 tmp, "-d", "0.3", "--width", str(W), "--height", str(H)] + extra,
                cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "src")),
            )
            assert proc.returncode == 0, proc.stdout[-3000:]
        names = os.listdir(tmp)
        assert any(n.endswith("_dis.png") for n in names), names
        assert any(n.endswith("_rec.png") for n in names), names


if __name__ == "__main__":
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print("PASS  {}".format(name))
        except Exception:
            import traceback

            print("FAIL  {}\n{}".format(name, traceback.format_exc()))
            failures.append(name)
    print("\n{}/{} passed".format(len(tests) - len(failures), len(tests)))
    raise SystemExit(1 if failures else 0)
