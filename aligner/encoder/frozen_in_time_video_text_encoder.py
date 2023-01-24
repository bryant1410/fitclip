import os
from typing import Iterable, Iterator

import torch
from overrides import overrides
from torchvision import transforms as T
from transformers import AutoTokenizer

from aligner.data.frame_sampler import FrameSampler, RandomFromUniformIntervalsFrameSampler, UniformFrameSampler
from aligner.encoder.frozen_in_time import FrozenInTime
from aligner.encoder.video_encoder import TYPE_TRANSFORM, TYPE_VIDEO_INPUT, float_standard_denormalize
from aligner.encoder.video_text_encoder import TYPE_TEXT_INPUT, TYPE_TOKENIZER, VideoTextEncoder
from aligner.transforms import ConvertBHWCtoBCHW, RandomResizedCropWithRandomInterpolation


def _normalize(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return t / torch.max(t.norm(dim=1, keepdim=True), eps * torch.ones_like(t))  # noqa


class FrozenInTimeVideoTextEncoder(VideoTextEncoder):
    # FIXME: set the max tokens by default as in CLIP, also to avoid spending too much memory when using prompts.
    def __init__(self, model: FrozenInTime, image_size: int = 224, num_frames: int = 4, max_tokens: int = 77) -> None:
        super().__init__()
        self.model = model

        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        self.tokenizer = AutoTokenizer.from_pretrained(model.text_params["model"])

        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        self.image_size = image_size
        self.num_frames = num_frames
        self.max_tokens = max_tokens

    @overrides(check_signature=False)
    def encode_video(self, video: TYPE_VIDEO_INPUT, eps: float = 1e-8) -> torch.Tensor:
        return _normalize(self.model.compute_video(video), eps=eps)

    @overrides(check_signature=False)
    def encode_text(self, text: TYPE_TEXT_INPUT, eps: float = 1e-8) -> torch.Tensor:
        return _normalize(self.model.compute_text(text), eps=eps)

    def _tokenize(self, texts: Iterable[str]) -> TYPE_TEXT_INPUT:
        texts = texts if isinstance(texts, (list, tuple)) else list(texts)
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_tokens)

    @overrides
    def get_tokenizer(self) -> TYPE_TOKENIZER:
        return self._tokenize

    @overrides
    def decode_text(self, text: TYPE_TEXT_INPUT) -> Iterator[str]:
        return self.tokenizer.batch_decode(text["input_ids"], skip_special_tokens=True)

    @overrides
    def get_train_frame_sampler(self) -> FrameSampler:
        return RandomFromUniformIntervalsFrameSampler(self.num_frames)

    @overrides
    def get_eval_frame_sampler(self) -> FrameSampler:
        return UniformFrameSampler(self.num_frames)

    @overrides
    def get_train_transform(self, dtype: torch.dtype) -> TYPE_TRANSFORM:
        return T.Compose([
            ConvertBHWCtoBCHW(),
            T.ConvertImageDtype(dtype),
            RandomResizedCropWithRandomInterpolation(self.image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            self.normalize,
        ])

    @overrides
    def get_eval_transform(self, dtype: torch.dtype) -> TYPE_TRANSFORM:
        return T.Compose([
            ConvertBHWCtoBCHW(),
            T.ConvertImageDtype(dtype),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            self.normalize,
        ])

    @property
    @overrides
    def should_pad_batch(self) -> bool:
        return True

    @overrides
    def to_bchw(self, t: torch.Tensor) -> torch.Tensor:
        return t

    @overrides
    def denormalize_video_tensor(self, video: TYPE_VIDEO_INPUT) -> torch.Tensor:
        return float_standard_denormalize(video, mean=self.normalize.mean, std=self.normalize.std)
