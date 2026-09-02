import argparse
import os

import torch
from PIL import Image

from models import ParametersEstimationModule
from core.config import Config
from core.checkpoint import checkTrainedRange, load_checkpoint
from datasets import FisheyeEffector, EvalDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch.cuda.is_available():", torch.cuda.is_available())

DEFAULTS = {
    "DATASET": {"HEIGHT": 256, "WIDTH": 512, "CROP": True},
    "MODEL": {"VGG_PRETRAINED": False, "FREEZE_VGG": False, "MAX_DISTORTION": 1.0},
    "TEST": {"CHECKPOINT": "checkpoint.pth.tar", "MAX_DISTORTION": 0.9},
}


def buildParser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", type=str, help="cfg/****.yml")
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="directory containing the fisheye images to rectify")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="defaults to <input_dir>/rectified")
    parser.add_argument("-ow", "--output_width", type=int, default=1280,
                        help="width of the rectified image")
    parser.add_argument("-oh", "--output_height", type=int, default=720,
                        help="height of the rectified image")
    parser.add_argument("--no_crop", dest="crop", action="store_false",
                        help="the input is a raw fisheye frame that was never cropped and "
                             "magnified (as a training render is); keep expansion at 1.0")
    parser.add_argument("--save-distortions", action="store_true",
                        help="also write <output_dir>/distortions.txt")
    return parser


def main(args):
    config = Config(args.config_file, defaults=DEFAULTS)

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError("No such directory: {}".format(args.input_dir))

    bound = float(config.get("MODEL.MAX_DISTORTION", 1.0))

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

    model_file = os.path.join("outputs", config.get("DATASET.NAME"), config.get("TEST.CHECKPOINT"))
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            'model_file "{}" does not exist -- run src/train.py first'.format(model_file)
        )
    raw = load_checkpoint(model_file, model, device=DEVICE)
    checkTrainedRange(raw, bound, "at predict time")

    # the network was trained at the dataset resolution, so feed it that
    dataset = EvalDataset(
        data_path=args.input_dir,
        transform=transform,
        height=config.get("DATASET.HEIGHT"),
        width=config.get("DATASET.WIDTH"),
    )
    if len(dataset) == 0:
        raise RuntimeError("no images found in {}".format(args.input_dir))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    output_dir = args.output_dir or os.path.join(args.input_dir, "rectified")
    os.makedirs(output_dir, exist_ok=True)

    predictions = []
    with torch.no_grad():
        for image, meta in dataloader:
            output = model(image.to(DEVICE)).reshape(-1).item()
            path = meta["path"][0]
            print("Got {:.5f} for {}".format(output, path))
            predictions.append((os.path.basename(path), output))

            # backward=True is the rectification direction; the previous code re-applied
            # the forward render and therefore distorted the image a second time
            effector = FisheyeEffector(
                height=args.output_height,
                width=args.output_width,
                distortion=output,
                backward=True,
                crop=args.crop,
            )
            full_resolution = Image.open(path).convert("RGB").resize(
                (args.output_width, args.output_height)
            )
            effector(full_resolution).save(os.path.join(output_dir, os.path.basename(path)))

    if args.save_distortions:
        with open(os.path.join(output_dir, "distortions.txt"), "w") as f:
            for name, value in predictions:
                f.write("{:.6f}\t{}\n".format(value, name))
        print("=> wrote {}/distortions.txt".format(output_dir))


if __name__ == "__main__":
    main(buildParser().parse_args())
