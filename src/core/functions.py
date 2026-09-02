import math
import random

import torch


def toDevice(batch, device):
    """Recursively move tensors of a (possibly nested) collated batch to ``device``."""
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: toDevice(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [toDevice(value, device) for value in batch]
        return type(batch)(moved) if isinstance(batch, tuple) else moved
    return batch


def train(model=None, dataloader=None, criterion=None, optimizer=None, device="cpu",
          log_interval=100):
    running_loss = 0.0
    num_iter = 0

    model = model.train()

    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(inputs.to(device))
        labels = toDevice(labels, device)
        loss = criterion(outputs, labels)
        value = loss.item()
        if not math.isfinite(value):
            # skip *before* backward(): stepping on a NaN loss would poison the weights
            # and Adam's moment estimates, from which they never recover
            print(
                "WARNING: non-finite loss at iteration {} (predicted distortion = {}); "
                "skipping this batch.".format(i + 1, outputs.detach().reshape(-1)[:4].tolist())
            )
            continue

        loss.backward()
        optimizer.step()

        running_loss += value
        num_iter += 1

        if log_interval and (i + 1) % log_interval == 0:
            print("Iter [{}/{}], Loss = {}".format(i + 1, len(dataloader), value))

    if num_iter == 0:
        return float("nan")
    return running_loss / num_iter


def val(model=None, dataloader=None, criterion=None, device="cpu"):
    running_loss = 0.0
    num_iter = 0

    model = model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            labels = toDevice(labels, device)
            loss = criterion(outputs, labels)

            value = loss.item()
            if not math.isfinite(value):
                continue
            running_loss += value
            num_iter += 1

    if num_iter == 0:
        if len(dataloader) == 0:
            print("WARNING: validation loader is empty, no validation loss available.")
        return float("nan")
    return running_loss / num_iter


def getDistortions(num_fragments=10, random_values=False, max_distortion=0.9,
                   symmetric=False):
    """The set of distortion parameters the effectors sample from.

    ``max_distortion`` bounds the range so a curriculum can actually grow the problem
    difficulty (it used to span ``[0, 0.9]`` for every fragment count, so switching
    stages only changed the granularity). ``symmetric`` adds the pincushion side,
    which the deterministic schedule never used to cover at all.
    """
    num_fragments = max(1, int(num_fragments))
    max_distortion = abs(float(max_distortion))
    low = -max_distortion if symmetric else 0.0

    if num_fragments == 1:
        return [low]

    interval = (max_distortion - low) / (num_fragments - 1)
    if random_values:
        return [low + interval * i + (random.random() - 0.5) * interval
                for i in range(num_fragments)]
    return [low + interval * i for i in range(num_fragments)]


def curriculumStage(epoch, switch_epoch, max_fragments=10):
    """(num_fragments, is_random) for a given 0-based epoch."""
    stage = int(epoch) // max(1, int(switch_epoch))
    num_fragments = min(int(max_fragments), stage + 2)
    return num_fragments, num_fragments >= int(max_fragments)


def curriculumDistortions(epoch, switch_epoch, max_distortion=0.9, start_distortion=0.3,
                          max_fragments=10):
    """Deterministic curriculum that grows range *and* granularity, then randomises.

    Returns the distortion list for the epoch. The stage only changes every
    ``switch_epoch`` epochs, so callers can rebuild the effectors just then.
    """
    num_fragments, is_random = curriculumStage(epoch, switch_epoch, max_fragments)
    if is_random:
        return getDistortions(num_fragments, random_values=True,
                              max_distortion=max_distortion, symmetric=True)
    # stage 0 sees a narrow range, the last deterministic stage the full one
    frac = (num_fragments - 2) / max(1, int(max_fragments) - 2)
    stage_max = start_distortion + (max_distortion - start_distortion) * frac
    return getDistortions(num_fragments, max_distortion=stage_max)
