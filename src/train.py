import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchsummary import summary

from models import ParametersEstimationModule
from core.functions import train, val, getDistortions
from core.losses import DistortionLoss
from core.config import Config
from datasets import DistortDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch.cuda.is_available():", torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument("config_file", type=str, help="cfg/****.yml")
parser.add_argument(
    "--workers",
    type=int,
    default=min(4, os.cpu_count() or 1),
    help="Number of DataLoader workers. Use 2-4 on Kaggle in most cases.",
)
parser.add_argument(
    "--no-summary",
    action="store_true",
    help="Skip the torchsummary forward pass.",
)


def _load_checkpoint_state(model, state_dict):
    """
    Load checkpoints produced both with and without DataParallel.
    This keeps old Kaggle checkpoints usable.
    """
    model_keys = list(model.state_dict().keys())
    state_keys = list(state_dict.keys())

    if not model_keys or not state_keys:
        model.load_state_dict(state_dict, strict=False)
        return

    model_has_module = model_keys[0].startswith("module.")
    state_has_module = state_keys[0].startswith("module.")

    if model_has_module and not state_has_module:
        state_dict = {
            "module." + key: value for key, value in state_dict.items()
        }
    elif not model_has_module and state_has_module:
        state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=False)


def main(args):
    config = Config(args.config_file).getDict()

    data_path = os.path.join("data", config["DATASET"]["NAME"])
    if not os.path.exists(data_path):
        print(
            "ERROR: No dataset named {}".format(
                config["DATASET"]["NAME"]
            )
        )
        exit(1)

    in_channels = 3

    model = ParametersEstimationModule(in_channels=in_channels).to(DEVICE)
    transform = model.getTransforms()

    # Kaggle normally exposes one GPU. DataParallel adds avoidable overhead
    # in that case, so only enable it when multiple GPUs are present.
    if DEVICE == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    if not args.no_summary:
        summary(
            model,
            input_size=(
                in_channels,
                config["DATASET"]["HEIGHT"],
                config["DATASET"]["WIDTH"],
            ),
        )

    criterion = DistortionLoss().to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["TRAIN"]["LEARNING_RATE"],
    )

    max_epoch = config["TRAIN"]["MAX_EPOCH"]
    last_epoch = 0
    train_losses = []
    val_losses = []

    enable_curriculum = config["TRAIN"]["CURRICULUM"]["ENABLED"]
    switch_epoch = config["TRAIN"]["CURRICULUM"]["SWITCH_EPOCH"]

    output_dir = os.path.join(
        "outputs",
        config["DATASET"]["NAME"],
    )
    model_state_file = os.path.join(
        output_dir,
        "checkpoint.pth.tar",
    )
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(model_state_file):
        checkpoint = torch.load(
            model_state_file,
            map_location=DEVICE,
        )
        last_epoch = checkpoint["epoch"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        _load_checkpoint_state(
            model,
            checkpoint["state_dict"],
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> load checkpoint (epoch {})".format(
                last_epoch
            )
        )

    if enable_curriculum:
        num_patterns = int(last_epoch / switch_epoch) + 2
        if num_patterns <= 10:
            distortions = getDistortions(
                num_patterns,
                random_values=False,
            )
        else:
            distortions = getDistortions(
                10,
                random_values=True,
            )
    else:
        distortions = getDistortions(
            10,
            random_values=True,
        )

    trainset = DistortDataset(
        list_path=os.path.join(data_path, "train.lst"),
        height=config["DATASET"]["HEIGHT"],
        width=config["DATASET"]["WIDTH"],
        transform=transform,
        distortions=distortions,
    )

    testset = DistortDataset(
        list_path=os.path.join(data_path, "val.lst"),
        height=config["DATASET"]["HEIGHT"],
        width=config["DATASET"]["WIDTH"],
        transform=transform,
        distortions=distortions,
    )

    # IMPORTANT: persistent_workers is intentionally NOT used.
    # updateEffector() changes the Dataset object every curriculum switch.
    # With persistent workers, old worker processes can retain stale
    # effectors. Respawning workers at each epoch keeps curriculum behavior
    # correct while still parallelizing image decoding/transforms.
    num_workers = max(0, args.workers)

    loader_common = {
        "batch_size": config["TRAIN"]["BATCH_SIZE"],
        "pin_memory": DEVICE == "cuda",
        "num_workers": num_workers,
    }

    if num_workers > 0:
        loader_common["prefetch_factor"] = 2

    trainloader = torch.utils.data.DataLoader(
        trainset,
        shuffle=True,
        **loader_common,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        shuffle=False,
        **loader_common,
    )

    for epoch in range(last_epoch, max_epoch):
        print("Epoch {}".format(epoch + 1))

        train_loss = train(
            model=model,
            dataloader=trainloader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        val_loss = val(
            model=model,
            dataloader=testloader,
            criterion=criterion,
            device=DEVICE,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            "Loss: train = {}, val = {}".format(
                train_loss,
                val_loss,
            )
        )

        plt.figure()
        plt.plot(
            range(1, len(train_losses) + 1),
            train_losses,
            label="train",
        )
        plt.plot(
            range(1, len(val_losses) + 1),
            val_losses,
            label="val",
        )
        plt.yscale("log")
        plt.legend()
        plt.savefig(
            os.path.join(output_dir, "losses.png")
        )
        plt.close()

        if (epoch + 1) % switch_epoch == 0:
            num_patterns = int(
                (epoch + 1) / switch_epoch
            ) + 2

            if enable_curriculum and num_patterns <= 10:
                distortions = getDistortions(
                    num_patterns,
                    random_values=False,
                )
            else:
                distortions = getDistortions(
                    10,
                    random_values=True,
                )

            trainset.updateEffector(
                distortions=distortions
            )
            testset.updateEffector(
                distortions=distortions
            )

        print(
            "=> saving checkpoint to {}".format(
                model_state_file
            )
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            model_state_file,
        )


if __name__ == "__main__":
    main(parser.parse_args())
