import itertools
import math
import sys
from typing import Generic, Iterable, Iterator, Literal, TypeVar, Union

from torch.utils.data import Sampler

T_co = TypeVar("T_co", covariant=True)


# We don't use `CycleIterator` from PyTorch Lightning because when used along with `itertools.islice`,
# it always creates a new iterator and wrongly starts from scratch because it's both an iterable and iterator (seems
# like the function calls `iter` internally).
class CycleSampler(Generic[T_co]):
    def __init__(self, data_source: Iterable[T_co], length: int = sys.maxsize) -> None:
        self.length = length
        self.data_source = data_source

    def __iter__(self) -> Iterator[T_co]:
        if not self.length:
            return

        counter = 0

        while True:
            it = iter(self.data_source)

            for elem in it:
                yield elem

                counter += 1

                if counter >= self.length:
                    return

    def __len__(self) -> int:
        return self.length


class RoundRobinMultiSourceSampler(Sampler[int]):
    """

    It supposes the dataset passed along to the `DataLoader` instance is a `ConcatDataset` instance.

    Recommended to use with `drop_last=True`.

    Some inspiration comes from the module `pytorch_lightning.trainer.supporters`.
    """

    def __init__(self, sub_samplers: Iterable[Iterable[int]], sequence_sizes: Union[int, Iterable[int]] = 1,
                 mode: Literal["min_size", "max_size_cycle"] = "min_size") -> None:
        sub_samplers = list(sub_samplers)
        sequence_sizes = list(sequence_sizes) if isinstance(sequence_sizes, Iterable) \
            else [sequence_sizes] * len(sub_samplers)

        assert len(sub_samplers) == len(sequence_sizes)
        assert all(len(sampler) for sampler in sub_samplers), ("All sub-samplers need to support `len` and be "  # noqa
                                                               "non-zero.")
        assert all(s > 0 for s in sequence_sizes)

        super().__init__(sub_samplers)

        self.sub_samplers = sub_samplers
        self.sequence_sizes = sequence_sizes
        self.mode = mode

        for sampler in self.sub_samplers:
            sampler._original_len = len(sampler)  # noqa

        if mode == "max_size_cycle":
            max_cycle, max_i = max((math.floor(cycle), - i) for i, cycle in enumerate(self._cycles()))
            max_i *= -1  # Trick to get the first sampler index among those of max cycle size.

            # Use a large number instead of the default inf because `len` can fail otherwise.
            # See https://stackoverflow.com/a/2481631/1165181
            self.sub_samplers = [sampler if i == max_i else CycleSampler(sampler, length=sys.maxsize)
                                 for i, sampler in enumerate(self.sub_samplers)]

            for i, sampler in enumerate(self.sub_samplers):
                if i != max_i:
                    sampler._original_len = len(sampler.data_source)  # noqa

    def _cycles(self) -> Iterator[float]:
        for sampler, seq_size in zip(self.sub_samplers, self.sequence_sizes):
            yield len(sampler) / seq_size

    def __iter__(self) -> Iterator[int]:
        iterators = [iter(sampler) for sampler in self.sub_samplers]
        while True:
            cum_size_in_concat_dataset = 0
            for it, size, sampler in zip(iterators, self.sequence_sizes, self.sub_samplers):
                i = -1
                for i, n in enumerate(itertools.islice(it, size)):
                    yield cum_size_in_concat_dataset + n
                if i < size - 1:
                    return
                cum_size_in_concat_dataset += sampler._original_len  # noqa

    def __len__(self) -> int:
        # Note in "max_size_cycle" mode the longest sampler will actually be the smallest one because the rest are
        # repeated infinitely.
        min_cycle, min_i = min((math.floor(cycle), i) for i, cycle in enumerate(self._cycles()))
        return (sum(seq_size * (min_cycle + int(i < min_i)) for i, seq_size in enumerate(self.sequence_sizes))
                + len(self.sub_samplers[min_i]) % self.sequence_sizes[min_i])
