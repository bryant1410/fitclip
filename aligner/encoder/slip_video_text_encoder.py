from typing import Iterable, Iterator, Union

import torch
from cached_path import cached_path
from overrides import overrides
from torchvision import transforms as T

from aligner.data.frame_sampler import FrameSampler, UniformFrameSampler
from aligner.encoder import slip
from aligner.encoder.slip import CLIP, SLIP, SimpleTokenizer
from aligner.encoder.video_encoder import TYPE_TRANSFORM, float_standard_denormalize
from aligner.encoder.video_text_encoder import TYPE_TEXT_INPUT, TYPE_TOKENIZER, TYPE_VIDEO_INPUT, VideoTextEncoder
from aligner.transforms import ConvertBHWCtoBCHW
from util.typing_utils import TYPE_PATH


def load_model(path: TYPE_PATH) -> Union[CLIP, SLIP]:
    checkpoint = torch.load(cached_path(path), map_location="cpu")
    args = checkpoint["args"]
    model = getattr(slip, args.model)(rand_embed=False, ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim)
    model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()})
    return model


class SlipVideoTextEncoder(VideoTextEncoder):
    def __init__(self, model: Union[CLIP, SLIP], num_frames: int = 4) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = SimpleTokenizer()
        self.num_frames = num_frames

        # Indirectly unregister the param as we don't use it and would otherwise give problems while training.
        if hasattr(self.model, "logit_scale"):
            delattr(self.model, "logit_scale")

    @overrides
    def encode_video(self, video: TYPE_VIDEO_INPUT) -> torch.Tensor:
        batch_size = video.shape[0]  # noqa

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

    def _tokenize(self, texts: Iterable[str]) -> TYPE_TEXT_INPUT:
        return {"input_ids": self.tokenizer(texts)}

    @overrides
    def get_tokenizer(self) -> TYPE_TOKENIZER:
        return self._tokenize

    @overrides
    def decode_text(self, text: TYPE_TEXT_INPUT) -> Iterator[str]:
        for text_instance in text:
            yield self.tokenizer.decode(text_instance["input_ids"])

    @overrides
    def get_train_frame_sampler(self) -> FrameSampler:
        raise NotImplementedError

    @overrides
    def get_eval_frame_sampler(self) -> FrameSampler:
        return UniformFrameSampler(self.num_frames)

    @overrides
    def get_train_transform(self, dtype: torch.dtype) -> TYPE_TRANSFORM:
        raise NotImplementedError

    @overrides
    def get_eval_transform(self, dtype: torch.dtype) -> TYPE_TRANSFORM:
        size = 224
        return T.Compose([
            ConvertBHWCtoBCHW(),
            T.ConvertImageDtype(dtype),
            T.Resize(size),
            T.CenterCrop(size),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
