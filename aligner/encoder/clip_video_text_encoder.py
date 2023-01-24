import os.path
import shutil
import tempfile
from typing import Iterable, Iterator, Tuple

import torch
from cached_path import cached_path
from clip import clip
from clip.model import CLIP
from overrides import overrides
from torch import nn
from torchvision import transforms as T

from aligner.data.frame_sampler import FrameSampler, RandomFromUniformIntervalsFrameSampler, UniformFrameSampler
from aligner.encoder.video_encoder import TYPE_TRANSFORM, float_standard_denormalize
from aligner.encoder.video_text_encoder import TYPE_TEXT_INPUT, TYPE_TOKENIZER, TYPE_VIDEO_INPUT, VideoTextEncoder
from aligner.transforms import ConvertBHWCtoBCHW, RandomResizedCropWithRandomInterpolation


# By default, `clip.load` uses part in half and part in single precision for GPU.
# But this may cause issues with the teacher-student model, and we can actually control it from the trainer.
def load_clip_in_float32(*args, **kwargs) -> Tuple[nn.Module, TYPE_TRANSFORM]:
    model, transform = clip.load(*args, **kwargs)
    model.float()
    return model, transform


# Necessary to use from Hydra so to get the first element of the tuple from `clip.load`.
# It also does more stuff than `clip.load`.
def load_clip_model(name: str, *args, **kwargs) -> nn.Module:
    temp_filepaths = []
    try:
        if "://" in name:
            name = cached_path(name)
        elif os.path.exists(name) and not os.path.isdir(name) and not os.path.isfile(name):
            # It could be a pipe. It could be created by a process substitution.
            # We copy it to a file because `clip.load` has a check that it's a file (and hence not a pipe).
            with tempfile.NamedTemporaryFile(delete=False) as output_file, open(name, "rb") as input_file:
                shutil.copyfileobj(input_file, output_file)
                name = output_file.name
                temp_filepaths.append(name)

        # We don't use the logic scale from CLIP but ours, so it may not exist. Here we need to re-create the variable,
        # so it doesn't fail when loading this `state_dict`.
        if os.path.exists(name):  # It doesn't apply if it's a model name.
            state_dict = torch.load(name)
            if "logit_scale" not in state_dict:
                state_dict["logit_scale"] = torch.tensor(float("nan"))
                with tempfile.NamedTemporaryFile(delete=False) as file:
                    # We create a new file to respect the original one.
                    torch.save(state_dict, file)
                    name = file.name
                temp_filepaths.append(name)

        if not args:  # If `args` is not empty, then `device` was set for `clip.load`.
            kwargs.setdefault("device", "cpu")  # To avoid bloating GPU 0 with each process' copy of it.

        return load_clip_in_float32(name, *args, **kwargs)[0]
    finally:
        for path in temp_filepaths:
            os.remove(path)


def _tokenize(texts: Iterable[str]) -> TYPE_TEXT_INPUT:
    return {"input_ids": clip.tokenize(texts, truncate=True)}  # noqa


class ClipVideoTextEncoder(VideoTextEncoder):
    def __init__(self, model: CLIP, num_frames: int = 4) -> None:
        super().__init__()
        self.model = model
        self.normalize = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.num_frames = num_frames

        # Indirectly unregister the param as we don't use it and would otherwise give problems while training.
        if hasattr(self.model, "logit_scale"):
            delattr(self.model, "logit_scale")

    @overrides
    def encode_video(self, video: TYPE_VIDEO_INPUT) -> torch.Tensor:
        batch_size = video.shape[0]

        images = video.view(-1, *video.shape[2:])
        encoded_video = self.model.encode_image(images)
        encoded_video = encoded_video / encoded_video.norm(dim=-1, keepdim=True)

        # Averaging the representations is the same as averaging the predictions:
        # <t, (i1+i2)/2> = 1/2 <t, i1+i2> = (<t, i1> + <t, i2>) / 2
        return encoded_video.view(batch_size, -1, *encoded_video.shape[1:]).mean(dim=1)

    @overrides
    def encode_text(self, text: TYPE_TEXT_INPUT) -> torch.Tensor:
        encoded_texts = self.model.encode_text(text["input_ids"])
        return encoded_texts / encoded_texts.norm(dim=-1, keepdim=True)

    @overrides
    def get_tokenizer(self) -> TYPE_TOKENIZER:
        return _tokenize

    @overrides
    def decode_text(self, text: TYPE_TEXT_INPUT) -> Iterator[str]:
        for text_instance in text:
            yield clip._tokenizer.decode(text_instance["input_ids"])

    @overrides
    def get_train_frame_sampler(self) -> FrameSampler:
        return RandomFromUniformIntervalsFrameSampler(self.num_frames)

    @overrides
    def get_eval_frame_sampler(self) -> FrameSampler:
        return UniformFrameSampler(self.num_frames)

    @overrides
    def get_train_transform(self, dtype: torch.dtype) -> TYPE_TRANSFORM:
        size = self.model.visual.input_resolution
        return T.Compose([
            ConvertBHWCtoBCHW(),
            T.ConvertImageDtype(dtype),
            RandomResizedCropWithRandomInterpolation(size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            self.normalize,
        ])

    @overrides
    def get_eval_transform(self, dtype: torch.dtype) -> TYPE_TRANSFORM:
        size = self.model.visual.input_resolution
        return T.Compose([
            ConvertBHWCtoBCHW(),
            T.ConvertImageDtype(dtype),
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
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
