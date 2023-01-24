import copy
from typing import Mapping, TypeVar

import torch
from torch import nn

T = TypeVar("T", bound=nn.Module)


def wise_state_dict(model1: T, model2: T, weight_for_2: float = 0.5) -> Mapping[str, torch.Tensor]:
    state_dict1 = dict(model1.named_parameters())
    state_dict2 = dict(model2.named_parameters())

    assert set(state_dict1) == set(state_dict2)

    return {k: (1 - weight_for_2) * state_dict1[k] + weight_for_2 * state_dict2[k] for k in state_dict1}


def wise(model1: T, model2: T, weight_for_2: float = 0.5, copy_model1: bool = True) -> T:
    assert type(model1) is type(model2)
    model = copy.deepcopy(model1 if copy_model1 else model2)
    model.load_state_dict(wise_state_dict(model1, model2, weight_for_2=weight_for_2))  # noqa
    return model
