from typing import Any, Mapping, Optional, Sequence, TypeVar, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.apply_func import apply_to_collection

T = TypeVar("T")


def pad(t: torch.Tensor, min_size: int, dim: int = 1, value: Any = 0) -> torch.Tensor:
    """Pads the dim `dim` in `t` with the value `value` so the size is at least `min_size`."""
    if dim < 0:
        dim += len(t.shape)

    if (count := t.shape[dim]) < min_size:
        # `pad` keyword arg goes from the last dim to the first one in pairs, where the first value of the pair is
        # for left padding and the other one for right padding.
        return F.pad(t, pad=(0, 0) * (len(t.shape) - 1 - dim) + (0, min_size - count), value=value)
    else:
        return t


def split_in_collection(data: T, split_size_or_sections: Union[int, Sequence[int]]) -> Sequence[T]:
    """Applies `split` to the inside tensors of the collections and also generates one collection for each of the
    returned elements from `split`."""
    type_ = type(data)
    if isinstance(data, torch.Tensor):
        return data.split(split_size_or_sections)
    elif isinstance(data, Mapping):
        zipped = zip(*(split_in_collection(v, split_size_or_sections) for v in data.values()))
        return [type_((k, v) for k, v in zip(data.keys(), z)) for z in zipped]
    elif isinstance(data, Sequence):
        return [type_(z) for z in zip(*(split_in_collection(e, split_size_or_sections) for e in data))]
    else:
        raise ValueError(f"Unsupported type for split: {type_}")


def _first_tensor_in_collection(data: Any) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, Mapping):
        return _first_tensor_in_collection(data.values())
    else:
        return _first_tensor_in_collection(next(iter(data)))


def all_gather(lightning_module: pl.LightningModule, data: Any, group: Optional[Any] = None,
               sync_grads: bool = False, return_world_size_dim: bool = False) -> Any:
    """Gathers a tensor, or multiple tensors inside a collection, so that the output number of dimensions is the same
    regardless of the accelerator.

    Note this is different from `pl.LightningModule.all_gather`, that for a single GPU it doesn't return a new
    dimension but for the parallel settings it does.
    """
    first_tensor_old_shape = _first_tensor_in_collection(data).shape
    output = lightning_module.all_gather(data, group=group, sync_grads=sync_grads)
    if len(first_tensor_new_shape := _first_tensor_in_collection(output).shape) == len(first_tensor_old_shape) + 1:
        return output if return_world_size_dim else apply_to_collection(output, torch.Tensor,
                                                                        lambda t: t.view(-1, *t.shape[2:]))
    elif len(first_tensor_new_shape) == len(first_tensor_old_shape):
        return apply_to_collection(output, torch.Tensor, torch.Tensor.unsqueeze, 0) if return_world_size_dim else output
    else:
        raise ValueError(f"Unexpected new shape for the first tensor in the collection: {first_tensor_new_shape} (old "
                         f"was {first_tensor_old_shape}). "
                         f"The new shape was expected to have the same number of dimensions or one more.")
