"""Useful utils when using `DataModuleStructuredGroup`."""
from typing import Any, Mapping, Sequence, Tuple

import torch

from aligner.video_text_module import TYPE_INPUT
from util.tensor_utils import pad

TYPE_MULTI_INPUT = Mapping[str, TYPE_INPUT]


# It's like `default_collate` but instead of a sequence we have a mapping, and we do `cat` instead of `stack`.
# It makes sense to be similar because we're merging multiple batches together.
# Note that using collate from the dataloader side. It's simpler, and more GPU-memory efficient.
def _cat_collate(batch: Sequence[Any]) -> Any:
    elem = batch[0]
    elem_type = type(batch)
    if isinstance(elem, torch.Tensor):
        return torch.cat(batch)  # noqa
    elif isinstance(elem, Mapping):
        return {k: _cat_collate([d[k] for d in batch]) for k in elem}
    elif isinstance(elem, (float, int, bytes, str)):
        return batch
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_cat_collate(samples) for samples in zip(*batch)))  # noqa
    elif isinstance(elem, Sequence):
        return [x for d in batch for x in d]
    else:
        raise TypeError(f"Not sure how to collate type {elem_type}")


def _merge_datasets_batch(batches_by_dataset: TYPE_MULTI_INPUT) -> Tuple[TYPE_INPUT, Sequence[int]]:
    lengths = [len(batch["video"]) for batch in batches_by_dataset.values()]

    max_text_len = max(batch["text"]["input_ids"].shape[-1] for batch in batches_by_dataset.values())
    for batch in batches_by_dataset.values():
        batch["text"] = {k: pad(v, min_size=max_text_len, dim=-1) for k, v in batch["text"].items()}

    batch = _cat_collate(list(batches_by_dataset.values()))

    return batch, lengths
