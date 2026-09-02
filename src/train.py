import argparse
import json
import os

import torch
import torch.optim as optim

from torchsummary import summary

from models import ParametersEstimationModule
from core.functions import train, val, curriculumDistortions, curriculumStage
from core.losses import DistortionLoss
from core.config import Config
from core.checkpoint import save_checkpoint, load_checkpoint
from datasets import DistortDataset

# Plotting must never be able to kill a run: the missing `plt` import used to raise
# NameError at the end of the first epoch, *before* the first torch.save, so no
# checkpoint was ever written. Backend is forced to Agg so headless boxes work too.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as error:  # pragma: no cover - optional dependency
    plt = None
    print("WARNING: matplotlib unavailable ({}); loss curves will be skipped.".format(error))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch.cuda.is_available():", torch.cuda.is_available())

DEFAULTS = {
    "DATASET": {"HEIGHT": 256, "WIDTH": 512},
    "MODEL": {"VGG_PRETRAINED": False, "FREEZE_VGG": False, "MAX_DISTORTION": 1.0},
    "TRAIN": {
        "MAX_EPOCH": 100,
        "BATCH_SIZE": 16,
        "LEARNING_RATE": 1e-4,
        "NUM_WORKERS": 1,
        "LOG_INTERVAL": 100,
        "CURRICULUM": {
            "ENABLED": True,
            "SWITCH_EPOCH": 30,
            "MAX_DISTORTION": 0.9,
            "START_DISTORTION": 0.3,
            "MAX_FRAGMENTS": 10,
        },
    },
}


def buildParser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", type=str, help="cfg/****.yml")
    parser.add_argument("--data_root", type=str, default="data", help="dataset root folder")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="defaults to outputs/<dataset>")
    parser.add_argument("--resume", action="store_true", help="continue from checkpoint")
    return parser


def main(args):
    config = Config(args.config_file, defaults=DEFAULTS)

    dataset = config.get("DATASET.NAME")
    data_path = os.path.join(args.data_root, dataset)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            "No dataset named '{}' at {} (run tools/prepare_cityscapes.py?)".format(
                dataset, data_path
            )
        )

    height, width = config.get("DATASET.HEIGHT"), config.get("DATASET.WIDTH")
    curriculum = config.get("TRAIN.CURRICULUM")
    # Two different ranges, and conflating them breaks training:
    #   MODEL.MAX_DISTORTION   physical limit of the parameterisation, bounds the network
    #                          output (same at train / test / predict time -- the
    #                          checkpoint records it and the scripts refuse a mismatch)
    #   CURRICULUM.MAX_DISTORTION  the largest |k| the *data* is rendered with; must stay
    #                          strictly inside the bound, since a squashed output can only
    #                          approach its own limit asymptotically
    bound = float(config.get("MODEL.MAX_DISTORTION", 1.0))
    data_max = float(curriculum.get("MAX_DISTORTION", min(0.9, 0.9 * bound)))
    if data_max >= bound:
        print("WARNING: TRAIN.CURRICULUM.MAX_DISTORTION ({}) >= MODEL.MAX_DISTORTION ({});"
              " targets at the bound are unreachable, using {}".format(
                  data_max, bound, 0.9 * bound))
        data_max = 0.9 * bound
    curriculum["MAX_DISTORTION"] = data_max
    in_channels = 3

    model = ParametersEstimationModule(
        in_channels=in_channels,
        vgg_pretrained=config.get("MODEL.VGG_PRETRAINED", False),
        freeze_vgg=config.get("MODEL.FREEZE_VGG", False),
        output_range=(-bound, bound),
    ).to(DEVICE)
    transform = model.getTransforms()

    if DEVICE == "cuda":
        model = torch.nn.DataParallel(model)

    summary(model, input_size=(in_channels, height, width), device=DEVICE)

    criterion = DistortionLoss(max_distortion=bound).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.get("TRAIN.LEARNING_RATE"))

    max_epoch = config.get("TRAIN.MAX_EPOCH")
    switch_epoch = max(1, int(curriculum.get("SWITCH_EPOCH", 30)))
    enable_curriculum = bool(curriculum.get("ENABLED", True))
    train_losses, val_losses = [], []
    last_epoch = 0

    output_dir = args.output_dir or os.path.join("outputs", dataset)
    os.makedirs(output_dir, exist_ok=True)
    model_state_file = os.path.join(output_dir, "checkpoint.pth.tar")

    if args.resume and os.path.exists(model_state_file):
        raw = load_checkpoint(model_state_file, model, optimizer=optimizer, device=DEVICE)
        last_epoch = int(raw.get("epoch", 0))
        train_losses = list(raw.get("train_losses", []))
        val_losses = list(raw.get("val_losses", []))
        print("=> resuming from epoch {}".format(last_epoch))

    def makeDataset(list_name, deterministic):
        return DistortDataset(
            list_path=os.path.join(data_path, list_name),
            height=height,
            width=width,
            transform=transform,
            distortions=[],
            deterministic=deterministic,
            crop=config.get("DATASET.CROP", True),
        )

    trainset = makeDataset("train.lst", deterministic=False)
    testset = makeDataset("val.lst", deterministic=True)
    if len(trainset) == 0 or len(testset) == 0:
        raise RuntimeError(
            "empty dataset ({} train / {} val images) -- check the .lst files in {}".format(
                len(trainset), len(testset), data_path
            )
        )

    common = {"batch_size": config.get("TRAIN.BATCH_SIZE"),
              "num_workers": config.get("TRAIN.NUM_WORKERS", 1)}
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, **common)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False, **common)

    current_stage = None
    for epoch in range(last_epoch, max_epoch):
        print("Epoch {}".format(epoch + 1))

        # curriculum: rebuild the effectors only when the stage actually changes
        stage = curriculumStage(epoch, switch_epoch, curriculum.get("MAX_FRAGMENTS", 10))
        if enable_curriculum and stage != current_stage:
            distortions = curriculumDistortions(
                epoch, switch_epoch,
                max_distortion=data_max,
                start_distortion=curriculum.get("START_DISTORTION", 0.3),
                max_fragments=curriculum.get("MAX_FRAGMENTS", 10),
            )
            print("  curriculum: {} fragments (random={}) -> {}".format(
                stage[0], stage[1], [round(d, 3) for d in distortions]))
            trainset.updateEffector(distortions=distortions)
            testset.updateEffector(distortions=distortions)
            current_stage = stage

        train_loss = train(
            model=model, dataloader=trainloader, criterion=criterion,
            optimizer=optimizer, device=DEVICE,
            log_interval=config.get("TRAIN.LOG_INTERVAL", 100),
        )
        val_loss = val(model=model, dataloader=testloader, criterion=criterion, device=DEVICE)
        print("Loss: train = {}, val = {}".format(train_loss, val_loss))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # save first: a plotting/IO problem must not cost an epoch of training
        print("=> saving checkpoint to {}".format(model_state_file))
        save_checkpoint(
            model_state_file, model, optimizer=optimizer,
            epoch=epoch + 1, train_losses=train_losses, val_losses=val_losses,
            model_max_distortion=bound, data_max_distortion=data_max,
            config=config.getDict(),
        )
        with open(os.path.join(output_dir, "losses.json"), "w") as f:
            json.dump({"train": train_losses, "val": val_losses}, f, indent=1)
        plotLosses(output_dir, train_losses, val_losses)

    print("done.")


def plotLosses(output_dir, train_losses, val_losses):
    if plt is None:
        return
    try:
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "losses.png"), dpi=90, bbox_inches="tight")
        plt.close()
    except Exception as error:  # pragma: no cover
        print("WARNING: could not write losses.png: {}".format(error))


if __name__ == "__main__":
    main(buildParser().parse_args())
