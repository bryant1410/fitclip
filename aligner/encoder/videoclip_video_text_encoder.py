import os
from typing import Iterable, Iterator, Optional

import torch
from overrides import overrides
from torchvision import transforms as T
from transformers import AutoTokenizer

from aligner.data.frame_sampler import ConsecutiveFrameSampler, FrameSampler
from aligner.encoder.s3dg import S3DG
from aligner.encoder.video_encoder import TYPE_TRANSFORM, TYPE_VIDEO_INPUT, float_standard_denormalize
from aligner.encoder.video_text_encoder import TYPE_TEXT_INPUT, TYPE_TOKENIZER, VideoTextEncoder
from aligner.encoder.videoclip import MMFusionSeparate
from aligner.transforms import ConvertBHWCtoCBHW, PadToMinFrames
from util.typing_utils import TYPE_PATH


class VideoClipVideoTextEncoder(VideoTextEncoder):
    def __init__(self, video_encoder_pretrained_path: Optional[TYPE_PATH] = None,
                 model_pretrained_path: Optional[TYPE_PATH] = None, num_frames: int = 32, max_tokens: int = 64) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.max_tokens = max_tokens

        self.video_encoder = S3DG()
        if video_encoder_pretrained_path:
            self.video_encoder.load_state_dict(torch.load(video_encoder_pretrained_path))

        self.model = MMFusionSeparate(max_video_len=num_frames)
        if model_pretrained_path:
            self.model.load_state_dict(torch.load(model_pretrained_path))

        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.model_name)

    @overrides(check_signature=False)
    def encode_video(self, video: TYPE_VIDEO_INPUT) -> torch.Tensor:
        batch_size, clip_count = video.shape[:2]
        assert batch_size == 1, "Only batch_size = 1 is supported for now."

        device = video.device

        # FIXME: VideoCLIP uses up to 32 clips per video, which complicates our implementation.
        #   These clips are randomly sampled when there's more than 32 clips.
        #   These clips are composed of non-overlapping 32 consecutive frames, and the video is sampled at 30 fps.
        video_features = self.video_encoder(video).view(batch_size, clip_count, self.video_encoder.output_size)
        video_mask = torch.ones((batch_size, self.num_frames), dtype=torch.bool, device=device)

        text = torch.tensor([[self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]], device=device) \
            .expand(batch_size, 2)
        text_mask = torch.ones((batch_size, 2), dtype=torch.bool, device=device)

        return self.model.forward_video(video_features, video_mask, text, text_mask)

    @overrides(check_signature=False)
    def encode_text(self, text: TYPE_TEXT_INPUT) -> torch.Tensor:
        return self.model.forward_text(text["input_ids"], text["attention_mask"])

    def _tokenize(self, texts: Iterable[str]) -> TYPE_TEXT_INPUT:
        texts = [f"{self.tokenizer.sep_token} {text}" for text in texts]
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_tokens)

    @overrides
    def get_tokenizer(self) -> TYPE_TOKENIZER:
        return self._tokenize

    @overrides
    def decode_text(self, text: TYPE_TEXT_INPUT) -> Iterator[str]:
        return self.tokenizer.batch_decode(text["input_ids"])

    @overrides
    def get_train_frame_sampler(self) -> FrameSampler:
        raise NotImplementedError

    @overrides
    def get_eval_frame_sampler(self) -> FrameSampler:
        return ConsecutiveFrameSampler(self.num_frames, fps=30)

    @overrides
    def get_train_transform(self, dtype: torch.dtype) -> TYPE_TRANSFORM:
        raise NotImplementedError

    @overrides
    def get_eval_transform(self, dtype: torch.dtype) -> TYPE_TRANSFORM:
        return T.Compose([
            ConvertBHWCtoCBHW(),
            T.ConvertImageDtype(dtype),
            T.Resize(224),
            T.CenterCrop(224),
            PadToMinFrames(self.num_frames, frame_dim=1),
        ])

    @property
    @overrides
    def should_pad_batch(self) -> bool:
        return False

    @overrides
    def to_bchw(self, t: torch.Tensor) -> torch.Tensor:
        return t.permute(0, 2, 1, 3, 4)

    @overrides
    def denormalize_video_tensor(self, video: TYPE_VIDEO_INPUT) -> torch.Tensor:
        return float_standard_denormalize(video)
