import collections
import itertools
from typing import Any, Iterable, Iterator, Literal, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


# See https://stackoverflow.com/a/50938015/1165181
def consume(it: Iterator[Any]) -> None:
    collections.deque(it, maxlen=0)


# See https://docs.python.org/3/library/itertools.html#itertools-recipes
def pairwise(it: Iterable[T]) -> Iterable[Tuple[T, T]]:
    a, b = itertools.tee(it)
    next(b, None)
    return zip(a, b)


# See https://stackoverflow.com/a/9001089/1165181
def can_be_iterated_more_than_once(it: Iterable[Any]) -> bool:
    try:
        object.__getattribute__(it, "__iter__")
    except AttributeError:
        return False
    try:
        object.__getattribute__(it, "__next__")
    except AttributeError:
        return True
    return False


# See `grouper` in https://docs.python.org/3/library/itertools.html#itertools-recipes.
def batch(iterable: Iterable[T], n: int, *, incomplete: Literal["fill", "ignore"] = "ignore",
          fill_value: Optional[Any] = None) -> Iterator[Iterable[T]]:
    """Batches the data into non-overlapping fixed-length batches.

    Examples:

    grouper("ABCDEFGH", 3) --> ABC DEF
    grouper("ABCDEFGH", 3, incomplete="fill", fill_value="x") --> ABC DEF GHx
    """
    args = [iter(iterable)] * n
    if incomplete == "fill":
        return itertools.zip_longest(*args, fillvalue=fill_value)
    elif incomplete == "ignore":
        return zip(*args)
    else:
        raise ValueError(f"Expected 'fill' or 'ignore'; got '{incomplete}'")


# See https://stackoverflow.com/a/312464/1165181
def batch_sequence(seq: Sequence[T], n: int) -> Iterator[Sequence[T]]:
    """Yield successive n-sized chunks from `seq`."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
