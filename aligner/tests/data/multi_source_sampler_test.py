import string
from typing import Literal

from torch.utils.data import ConcatDataset, DataLoader, SequentialSampler

from aligner.data.multi_source_sampler import RoundRobinMultiSourceSampler


def _create_sample_data_loader(mode: Literal["min_size", "max_size_cycle"]) -> DataLoader:
    dataset1 = string.ascii_lowercase
    dataset2 = range(10)
    dataset = ConcatDataset([dataset1, dataset2])  # noqa
    sampler = RoundRobinMultiSourceSampler([SequentialSampler(dataset1), SequentialSampler(dataset2)],
                                           sequence_sizes=[4, 3], mode=mode)
    return DataLoader(dataset, sampler=sampler, batch_size=None)


def test_multi_source_sampler_min_size() -> None:
    data_loader = _create_sample_data_loader(mode="min_size")

    expected_list = ["a", "b", "c", "d", 0, 1, 2, "e", "f", "g", "h", 3, 4, 5, "i", "j", "k", "l", 6, 7, 8, "m", "n",
                     "o", "p", 9]
    assert len(data_loader) == len(expected_list)
    assert list(data_loader) == expected_list


def test_multi_source_sampler_max_size_cycle() -> None:
    data_loader = _create_sample_data_loader(mode="max_size_cycle")

    expected_list = ["a", "b", "c", "d", 0, 1, 2, "e", "f", "g", "h", 3, 4, 5, "i", "j", "k", "l", 6, 7, 8, "m", "n",
                     "o", "p", 9, 0, 1, "q", "r", "s", "t", 2, 3, 4, "u", "v", "w", "x", 5, 6, 7, "y", "z"]
    assert len(data_loader) == len(expected_list)
    assert list(data_loader) == expected_list
