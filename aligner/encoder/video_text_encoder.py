from abc import abstractmethod
from typing import Callable, Iterable, Iterator, Mapping, Tuple

import torch
from overrides import overrides

from aligner.encoder.video_encoder import TYPE_VIDEO_INPUT, VideoEncoder

TYPE_TEXT_INPUT = Mapping[str, torch.Tensor]
TYPE_OUTPUT = Tuple[torch.Tensor, torch.Tensor]

TYPE_TOKENIZER = Callable[[Iterable[str]], Mapping[str, torch.Tensor]]


class VideoTextEncoder(VideoEncoder):
    @abstractmethod
    def encode_text(self, text: TYPE_TEXT_INPUT) -> torch.Tensor:
        raise NotImplementedError

    @overrides(check_signature=False)
    def forward(self, video: TYPE_VIDEO_INPUT, text: TYPE_TEXT_INPUT) -> TYPE_OUTPUT:
        return self.encode_video(video), self.encode_text(text)

    @abstractmethod
    def get_tokenizer(self) -> TYPE_TOKENIZER:
        raise NotImplementedError

    @abstractmethod
    def decode_text(self, text: TYPE_TEXT_INPUT) -> Iterator[str]:
        """Decodes a batch of texts."""
        raise NotImplementedError
