import argparse
import os

import torch
import torchvision.transforms as transforms

from models import ParametersEstimationModule
from core.functions import getDistortions
from core.losses import DistortionLoss
from core.config import Config
from core.checkpoint import checkTrainedRange, load_checkpoint
from datasets import DistortDataset, FisheyeEffector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch.cuda.is_available():", torch.cuda.is_available())

DEFAULTS = {
    "DATASET": {"HEIGHT": 256, "WIDTH": 512},
    "MODEL": {"VGG_PRETRAINED": False, "FREEZE_VGG": False, "MAX_DISTORTION": 1.0},
    "TEST": {
        "CHECKPOINT": "checkpoint.pth.tar",
        "SAVE_RESULTS": True,
        "OUTPUT_DIR": "results",
        "NUM_FRAGMENTS": 10,
        "MAX_DISTORTION": 0.9,
        "OUTPUT_SIZE": {"HEIGHT": 256, "WIDTH": 512},
    },
}


def buildParser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", type=str, help="cfg/****.yml")
    parser.add_argument("--data_root", type=str, default="data", help="dataset root folder")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="defaults to TEST.OUTPUT_DIR (relative to the repo root)")
    parser.add_argument("--split", type=str, default="test.lst")
    return parser


def main(args):
    config = Config(args.config_file, defaults=DEFAULTS)

    dataset = config.get("DATASET.NAME")
    data_path = os.path.join(args.data_root, dataset)
    list_path = os.path.join(data_path, args.split)
    if not os.path.isfile(list_path):
        raise FileNotFoundError("no {} (run tools/prepare_cityscapes.py?)".format(list_path))

    height, width = config.get("DATASET.HEIGHT"), config.get("DATASET.WIDTH")
    # the *bound* has to match training (it is baked into the weights' meaning); the
    # evaluation range is a property of the benchmark, and must stay inside the bound
    bound = float(config.get("MODEL.MAX_DISTORTION", 1.0))
    max_distortion = float(config.get("TEST.MAX_DISTORTION", min(0.9, 0.9 * bound)))
    if max_distortion >= bound:
        print("WARNING: TEST.MAX_DISTORTION ({}) >= MODEL.MAX_DISTORTION ({}); using {}"
              .format(max_distortion, bound, 0.9 * bound))
        max_distortion = 0.9 * bound

    model = ParametersEstimationModule(
        in_channels=3,
        vgg_pretrained=config.get("MODEL.VGG_PRETRAINED", False),
        freeze_vgg=config.get("MODEL.FREEZE_VGG", False),
        output_range=(-bound, bound),
    ).to(DEVICE)
    model.eval()
    transform = model.getTransforms()

    if DEVICE == "cuda":
        model = torch.nn.DataParallel(model)

    model_file = os.path.join("outputs", dataset, config.get("TEST.CHECKPOINT"))
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            'model_file "{}" does not exist -- run src/train.py first'.format(model_file)
        )
    raw = load_checkpoint(model_file, model, device=DEVICE)
    checkTrainedRange(raw, bound, "at test time")

    # evaluate over the full symmetric range, not just the positive half the old
    # getDistortions(11) produced
    distortions = getDistortions(
        config.get("TEST.NUM_FRAGMENTS", 10), max_distortion=max_distortion, symmetric=True
    )
    output_dir = args.output_dir or (
        config.get("TEST.OUTPUT_DIR") if config.get("TEST.SAVE_RESULTS") else None
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # deterministic=True pins which effector each image gets, so the run is reproducible
    crop = config.get("DATASET.CROP", True)
    dataset_ = DistortDataset(
        list_path=list_path,
        height=height,
        width=width,
        transform=transform,
        distortions=distortions,
        deterministic=True,
        crop=crop,
        output_dir=output_dir,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset_, batch_size=1, shuffle=False, num_workers=config.get("TRAIN.NUM_WORKERS", 1)
    )

    criterion = DistortionLoss(max_distortion=bound).to(DEVICE)
    out_height = config.get("TEST.OUTPUT_SIZE.HEIGHT", height)
    out_width = config.get("TEST.OUTPUT_SIZE.WIDTH", width)

    rows = []
    with torch.no_grad():
        for idx, (data, label) in enumerate(dataloader):
            prediction = model(data.to(DEVICE)).reshape(-1)
            gt = label["distortion"].reshape(-1)
            error = abs(prediction.item() - gt.item())
            coords = label["coords"].to(DEVICE)
            expansion = label["expansion"].to(DEVICE)
            # the same geometric relation, evaluated at the prediction and at the truth
            relation_pred = criterion(prediction, (coords, expansion)).item()
            relation_gt = criterion(gt, (coords, expansion)).item()
            print("image {:>5}: got {:>8.4f}, distorted by {:>8.4f} "
                  "| abs err {:>7.4f} | relation err {:.3e} vs {:.3e}".format(
                      idx, prediction.item(), gt.item(), error, relation_pred, relation_gt))
            rows.append((idx, label["path"][0], gt.item(), prediction.item(), error,
                         relation_pred, relation_gt))

            if output_dir:
                # rectify: the *inverse* direction of the render, not the render again
                effector = FisheyeEffector(
                    height=out_height, width=out_width,
                    distortion=prediction.item(), backward=True, crop=crop,
                )
                image = transforms.ToPILImage(mode="RGB")(ParametersEstimationModule.denormalize(data[0]))
                if image.size != (out_width, out_height):
                    # TEST.OUTPUT_SIZE is the size of the *deliverable*; resample it here
                    # so an upsampled output is interpolated by PIL instead of reaching the
                    # effector at the dataset size (the effector's own fallback is nearest,
                    # which looks blocky when magnifying).
                    image = image.resize((out_width, out_height))
                effector(image).save(os.path.join(output_dir, "{}_rec.jpg".format(idx)))

    if rows:
        errors = [r[4] for r in rows]
        print("\n{} images | distortion MAE = {:.4f} | median {:.4f} | max {:.4f}".format(
            len(rows), sum(errors) / len(errors), sorted(errors)[len(errors) // 2], max(errors)))
        print("mean relation error: predicted {:.3e} vs ground truth {:.3e}".format(
            sum(r[5] for r in rows) / len(rows), sum(r[6] for r in rows) / len(rows)))
    else:
        print("WARNING: nothing to evaluate, {} is empty".format(list_path))

    if output_dir:
        with open(os.path.join(output_dir, "metrics.csv"), "w") as f:
            f.write("idx,path,gt,prediction,abs_err,relation_err_pred,relation_err_gt\n")
            for idx, path, gt, pred, err, rp, rg in rows:
                f.write('{},{},{:.6f},{:.6f},{:.6f},{:.6e},{:.6e}\n'.format(idx, path, gt, pred, err, rp, rg))
        print("=> wrote {}/metrics.csv".format(output_dir))


if __name__ == "__main__":
    main(buildParser().parse_args())
