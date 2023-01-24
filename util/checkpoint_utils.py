from typing import MutableMapping

import torch
from cached_path import cached_path

from util.typing_utils import TYPE_PATH


def state_dict_from_checkpoint_path(checkpoint_path: TYPE_PATH, prefix: str = "") -> MutableMapping[str, torch.Tensor]:
    prefix += ("" if prefix.endswith(".") or not prefix else ".")
    checkpoint = torch.load(cached_path(checkpoint_path))
    return {k[len(prefix):]: v for k, v in checkpoint["state_dict"].items() if k.startswith(prefix)}
