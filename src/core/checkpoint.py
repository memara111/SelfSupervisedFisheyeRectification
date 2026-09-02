"""Checkpoint save/load that cannot silently do nothing.

``load_state_dict(..., strict=False)`` (what the scripts used to do) silently loads zero
tensors when the ``module.`` prefix of a ``DataParallel`` checkpoint disagrees with a
plain model -- e.g. train on one device setup and test on the other. The weights stay
random, nothing is raised, and the reported metrics are meaningless. Keys are
normalised here and mismatches are reported.
"""

import torch

PREFIX = "module."


def _normalize(state_dict, want_prefix):
    stripped = {
        (key[len(PREFIX):] if key.startswith(PREFIX) else key): value
        for key, value in state_dict.items()
    }
    if not want_prefix:
        return stripped
    return {PREFIX + key: value for key, value in stripped.items()}


def save_checkpoint(path, model, optimizer=None, **metadata):
    payload = dict(metadata)
    payload["state_dict"] = model.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def checkTrainedRange(raw, max_distortion, context=""):
    """Guard against evaluating a model with a different output bound than it was trained
    with: the bound is part of what the weights mean (it is applied after the squashing),
    so a mismatch silently rescales every prediction and every saved rectification.

    Checkpoints that predate the field are accepted (stored is None)."""
    stored = raw.get("model_max_distortion", raw.get("max_distortion")) \
        if isinstance(raw, dict) else None
    if stored is None or abs(float(stored) - float(max_distortion)) <= 1e-9:
        return
    raise RuntimeError(
        "MODEL.MAX_DISTORTION mismatch{}: checkpoint was trained with {}, config says {}. "
        "Set MODEL.MAX_DISTORTION: {} for train / test / predict.".format(
            " " + context if context else "", stored, max_distortion, stored))


def load_checkpoint(path, model, optimizer=None, device="cpu", strict=True):
    """Load ``path`` into ``model`` (wrapping or not wrapping in DataParallel is fine).

    Returns the checkpoint dict so callers can read ``epoch`` / loss history from it.
    """
    map_location = torch.device(device) if isinstance(device, str) else device
    raw = torch.load(path, map_location=map_location)
    is_wrapper = isinstance(model, torch.nn.DataParallel)

    if isinstance(raw, dict) and "state_dict" in raw:
        state = _normalize(raw["state_dict"], is_wrapper)
    else:  # a bare state_dict
        state = _normalize(raw, is_wrapper)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        message = "checkpoint keys do not match the model: {} missing, {} unexpected".format(
            list(missing)[:4], list(unexpected)[:4]
        )
        if strict:
            raise RuntimeError(
                "Refusing to continue with partially loaded weights ({}). "
                "Pass strict=False only if you know why.".format(message)
            )
        print("WARNING: " + message)
    else:
        print("=> loaded {} (all keys matched)".format(path))

    if optimizer is not None and isinstance(raw, dict) and raw.get("optimizer"):
        optimizer.load_state_dict(raw["optimizer"])
    return raw if isinstance(raw, dict) else {}
