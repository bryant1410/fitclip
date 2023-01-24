import collections.abc
import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import torch
from overrides import overrides
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from aligner.data.frame_sampler import FrameSampler
from aligner.data.video_reader import VideoReader
from aligner.encoder.video_encoder import TYPE_TRANSFORM
from util.typing_utils import TYPE_PATH

T = TypeVar("T")

LOGGER = logging.getLogger(__name__)


def get_filename_without_extension(path: TYPE_PATH) -> str:
    return os.path.basename(path).split(".", maxsplit=1)[0]


# TODO: support taking multiple clips per video, where they are chosen according to some strategy.
class VideoDataset(Dataset, Generic[T], ABC):
    def __init__(self, video_paths: Iterable[TYPE_PATH], frame_sampler: Union[FrameSampler, Mapping[str, FrameSampler]],
                 transform: Union[TYPE_TRANSFORM, Mapping[str, TYPE_TRANSFORM]] = lambda x: x,
                 video_key_name: str = "video", target_key_name: str = "target", pad_batch: bool = True,
                 cache: bool = False) -> None:
        super().__init__()
        self.video_paths = video_paths if hasattr(video_paths, "__getitem__") else list(video_paths)
        self.target_key_name = target_key_name
        self.pad_batch = pad_batch
        self.cache = cache

        if isinstance(frame_sampler, Mapping):
            self.frame_sampler_map = {f"{video_key_name}_{k}": v for k, v in frame_sampler.items()}
        else:
            self.frame_sampler_map = {video_key_name: frame_sampler}

        if isinstance(transform, Mapping):
            self.transform_map = {f"{video_key_name}_{k}": v for k, v in transform.items()}
        else:
            self.transform_map = {video_key_name: transform}

        if set(self.frame_sampler_map) != set(self.transform_map):
            if video_key_name in self.frame_sampler_map:
                self.frame_sampler_map = {k: self.frame_sampler_map[video_key_name] for k in self.transform_map}
            elif video_key_name in self.transform_map:
                self.transform_map = {k: self.transform_map[video_key_name] for k in self.frame_sampler_map}
            else:
                raise ValueError("The provided keys for the frame sampler and the transform don't match.")

    @abstractmethod
    def _get_target(self, video_idx: int) -> T:
        """Returns the target associated with `self.video_paths[video_idx]`."""
        raise NotImplementedError

    @functools.lru_cache
    def _get_video_id(self, video_idx: int) -> str:
        return get_filename_without_extension(self.video_paths[video_idx])

    def _get_times(self, video_idx: int) -> Tuple[Optional[float], Optional[float]]:
        """Returns the video clip start and end times for the given video index, if any."""
        return None, None

    @functools.lru_cache(maxsize=None)
    def _cached_get_item(self, video_idx: int) -> Mapping[str, Union[torch.Tensor, str, T]]:
        path = self.video_paths[video_idx]
        video_id = self._get_video_id(video_idx)
        video_reader = VideoReader.from_path(path)

        start_time, end_time = self._get_times(video_idx)

        start_frame_idx = 0 if start_time is None else video_reader.time_to_indices(start_time).item()
        end_frame_idx = len(video_reader) - 1 if end_time is None else video_reader.time_to_indices(end_time).item()

        idxs_map = {k: frame_sampler(start_frame_idx, end_frame_idx, fps=video_reader.get_avg_fps())
                    for k, frame_sampler in self.frame_sampler_map.items()}

        frames_map = {k: video_reader(idxs) for k, idxs in idxs_map.items()}

        return {
            self.target_key_name: self._get_target(video_idx),
            "video_id": video_id,
            **{k: transform(frames_map[k]) for k, transform in self.transform_map.items()},
        }

    @overrides
    def __getitem__(self, video_idx: int) -> Mapping[str, Union[torch.Tensor, str, T]]:
        # Note we have to explicitly pass `self` to the wrapped one.
        fn = self._cached_get_item if self.cache else functools.partial(self._cached_get_item.__wrapped__, self)  # noqa
        return fn(video_idx)

    def __len__(self) -> int:
        return len(self.video_paths)

    def _collate(self, batch: Sequence[Any]) -> Any:
        if self.pad_batch:
            elem = batch[0]
            if isinstance(elem, torch.Tensor):
                return pad_sequence(batch, batch_first=True)  # noqa
            elif isinstance(elem, collections.abc.Mapping):
                return {k: self._collate([d[k] for d in batch]) if k in self.transform_map
                        else default_collate([d[k] for d in batch])
                        for k in elem}

        return default_collate(batch)

    def collate(self, batch: Sequence[Any]) -> Any:
        # Use an auxiliary function instead of doing it directly here because it's recursive, and it may also be
        # overridden. so in the recursion the overridden version may be called instead of this one.
        return self._collate(batch)
