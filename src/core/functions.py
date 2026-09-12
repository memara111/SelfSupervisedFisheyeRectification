import random
import torch


def train(
    model=None,
    dataloader=None,
    criterion=None,
    optimizer=None,
    device="cpu",
):
    running_loss = 0.0
    num_iter = 0
    max_iter = len(dataloader)

    model = model.train()

    for i, (inputs, labels) in enumerate(dataloader):
        # set_to_none=True avoids an unnecessary fill operation.
        optimizer.zero_grad(set_to_none=True)

        # DataLoader uses pinned memory on CUDA, so non_blocking=True can
        # overlap host->device copies with work more effectively.
        inputs = inputs.to(
            device,
            non_blocking=(device == "cuda"),
        )

        labels = [
            (
                x.to(device, non_blocking=(device == "cuda")),
                y.to(device, non_blocking=(device == "cuda")),
            )
            for x, y in labels
        ]

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.detach().item()
        num_iter += 1

        if (i + 1) % 100 == 0:
            print(
                "Iter [{}/{}], Loss = {}".format(
                    i + 1,
                    max_iter,
                    running_loss / num_iter,
                )
            )

    return running_loss / num_iter


def val(
    model=None,
    dataloader=None,
    criterion=None,
    device="cpu",
):
    running_loss = 0.0
    num_iter = 0

    model = model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(
                device,
                non_blocking=(device == "cuda"),
            )

            labels = [
                (
                    x.to(device, non_blocking=(device == "cuda")),
                    y.to(device, non_blocking=(device == "cuda")),
                )
                for x, y in labels
            ]

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.detach().item()
            num_iter += 1

    return running_loss / num_iter


def getDistortions(num_fragments=10, random_values=False):
    interval = 0.9 / (num_fragments - 1)
    distortions = [interval * i for i in range(num_fragments)]

    if random_values:
        distortions = [
            val - random.random() * interval
            for val in distortions
        ]

    return distortions
