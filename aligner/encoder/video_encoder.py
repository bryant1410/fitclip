from abc import abstractmethod
from typing import Callable, Optional, Tuple

import torch
from overrides import overrides
from torch import nn

from aligner.data.frame_sampler import FrameSampler

TYPE_VIDEO_INPUT = torch.Tensor
TYPE_TRANSFORM = Callable[[torch.Tensor], torch.Tensor]


class VideoEncoder(nn.Module):
    @abstractmethod
    def encode_video(self, video: TYPE_VIDEO_INPUT) -> torch.Tensor:
        raise NotImplementedError

    @overrides(check_signature=False)
    def forward(self, video: TYPE_VIDEO_INPUT) -> torch.Tensor:
        return self.encode_video(video)

    @abstractmethod
    def get_train_frame_sampler(self) -> FrameSampler:
        raise NotImplementedError

    @abstractmethod
    def get_eval_frame_sampler(self) -> FrameSampler:
        raise NotImplementedError

    @abstractmethod
    def get_train_transform(self, dtype: torch.dtype) -> TYPE_TRANSFORM:
        raise NotImplementedError

    @abstractmethod
    def get_eval_transform(self, dtype: torch.dtype) -> TYPE_TRANSFORM:
        raise NotImplementedError

    @property
    # Don't set as abstract method to avoid some boilerplate in subclasses.
    # See https://stackoverflow.com/a/42529760/1165181
    def should_pad_batch(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_bchw(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def denormalize_video_tensor(self, video: TYPE_VIDEO_INPUT) -> torch.Tensor:
        """Converts a transformed video tensor into an unsigned 8-bit integer tensor in the range 0-255."""
        raise NotImplementedError


def float_standard_denormalize(video: TYPE_VIDEO_INPUT, mean: Optional[Tuple[float, float, float]] = None,
                               std: Optional[Tuple[float, float, float]] = None) -> torch.Tensor:
    if std is not None:
        video *= torch.tensor(std, device=video.device, dtype=video.dtype).view(-1, 1, 1)

    if mean is not None:
        video += torch.tensor(mean, device=video.device, dtype=video.dtype).view(-1, 1, 1)

    return (video * 255).to(torch.uint8)  # noqa
