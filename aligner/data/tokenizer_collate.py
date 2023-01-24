import collections.abc
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple, Union

from overrides import overrides
from pytorch_lightning.utilities.apply_func import apply_to_collection
from torch.utils.data.dataloader import default_collate

from aligner.encoder.video_text_encoder import TYPE_TOKENIZER


# Derived from `default_collate`.
def batch_tokenize_collate(batch: Sequence[Any], tokenizer: TYPE_TOKENIZER) -> Any:
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, (str, bytes)):
        return tokenizer(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {k: batch_tokenize_collate([d[k] for d in batch], tokenizer) for k in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(batch_tokenize_collate(samples, tokenizer) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("Each element in sequence of batch should be of equal size.")
        transposed = zip(*batch)
        return [batch_tokenize_collate(samples, tokenizer) for samples in transposed]
    else:
        raise TypeError(f"Batch must contain strings, mappings or sequences; found {elem_type}.")


class TokenizerCollate(ABC):
    """`DataLoader` collate function that batch-tokenizes part of the batch.

    The pros of batch-tokenizing during collation are:
    1) We can pad at the same time, based on the longest sequence. If we tokenized in the dataset, we wouldn't know
    what size to take, and we may take a long one, wasting computing and especially memory. If we batch-tokenize when
    iterating through the data_module loader, we are in the main thread and wasting valuable time that could be used for
    the GPU.
    2) The `tokenizers` library is written in Rust and may have some optimizations for batch-tokenizing (apart from
    multi-threading, which is disabled so each data_module loader worker uses one CPU core.)
    """

    def __init__(self, tokenizer: Union[TYPE_TOKENIZER, Mapping[str, TYPE_TOKENIZER]], *,
                 batch_tokenize_collate_fn: Callable[[Sequence[Any], TYPE_TOKENIZER], Any] = batch_tokenize_collate,
                 default_collate_fn: Callable[[Sequence[Any]], Any] = default_collate) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_tokenize_collate_fn = batch_tokenize_collate_fn
        self.default_collate_fn = default_collate_fn

    @abstractmethod
    def _split_uncollated_batch(self, batch: Sequence[Any]) -> Tuple[Sequence[Any], Sequence[Any]]:
        """Splits the batch into a pair where the first element is going to be processed with the default collate
        function and each of the elements in the second one are going to be batch-tokenized."""
        raise NotImplementedError

    @abstractmethod
    def _join_collated_batch(self, collated_with_default: Any, collated_with_tokenizer: Any) -> Any:
        raise NotImplementedError

    def __call__(self, batch: Sequence[Any]) -> Any:
        s1, s2 = self._split_uncollated_batch(batch)
        batch_tokenized = apply_to_collection(self.tokenizer, Callable, lambda t: self.batch_tokenize_collate_fn(s2, t))
        return self._join_collated_batch(self.default_collate_fn(s1), batch_tokenized)


class MappingTokenizerCollate(TokenizerCollate):
    def __init__(self, tokenizer: TYPE_TOKENIZER, keys_to_tokenize: Union[str, Iterable[str]], **kwargs) -> None:
        super().__init__(tokenizer, **kwargs)
        self.keys_to_tokenize = frozenset({keys_to_tokenize} if isinstance(keys_to_tokenize, str) else keys_to_tokenize)

    @overrides(check_signature=False)
    def _split_uncollated_batch(self,
                                batch: Sequence[Mapping[str, Any]]) -> Tuple[Sequence[Any], Sequence[Any]]:
        return [{k: v for k, v in d.items() if k not in self.keys_to_tokenize} for d in batch], \
               [{k: v for k, v in d.items() if k in self.keys_to_tokenize} for d in batch]

    @overrides(check_signature=False)
    def _join_collated_batch(self, collated_with_default: Any, collated_with_tokenizer: Any) -> Any:
        # If the tokenizer is actually composed of many tokenizers, we flatten out the structure.
        if isinstance(self.tokenizer, Mapping):
            collated_with_tokenizer = {f"{k_child}_{k_parent}": v_child
                                       for k_parent, v_parent in collated_with_tokenizer.items()
                                       for k_child, v_child in v_parent.items()}

        return {**collated_with_default, **collated_with_tokenizer}
