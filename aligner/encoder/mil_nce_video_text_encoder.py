import re
from typing import Any, Iterable, Iterator, Mapping, Optional, Union

import numpy as np
import torch
from cached_path import cached_path
from overrides import overrides
from torch import nn
from torchvision import transforms as T

from aligner.data.frame_sampler import ConsecutiveFrameSampler, FrameSampler
from aligner.encoder.s3dg import S3DG
from aligner.encoder.video_encoder import TYPE_TRANSFORM, float_standard_denormalize
from aligner.encoder.video_text_encoder import TYPE_TEXT_INPUT, TYPE_TOKENIZER, TYPE_VIDEO_INPUT, VideoTextEncoder
from aligner.transforms import ConvertBHWCtoCBHW, PadToMinFrames
from util.typing_utils import TYPE_PATH


def load_pretrained_video_encoder(path: TYPE_PATH,
                                  map_location: Optional[Union[str, torch.device]] = None) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = get_video_encoder_state_dict_from_pretrained_mil_nce_checkpoint(checkpoint) \
        if "state_dict" in checkpoint else checkpoint

    # Backward compatibility, also with the MIL-NCE paper pretrained one.
    return {k: v for k, v in state_dict.items() if not k.startswith("text_module.")}


def load_pretrained_text_encoder(path: TYPE_PATH,
                                 map_location: Optional[Union[str, torch.device]] = None) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if "state_dict" in checkpoint:
        return get_text_encoder_state_dict_from_pretrained_mil_nce_checkpoint(checkpoint)
    elif any(k.startswith("text_module.") for k in checkpoint):
        # Backward compatibility, also with a MIL-NCE paper pretrained one.
        prefix = "text_module."
        return {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
    else:
        return checkpoint


def get_video_encoder_state_dict_from_pretrained_mil_nce_checkpoint(
        checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    pl_module_state_dict = checkpoint["state_dict"]

    # Look for the corresponding encoder, with backward compatibility.
    prefix = "encoder." if any(k.startswith("encoder.") for k in pl_module_state_dict.keys()) else "video_encoder."
    return {k[len(prefix):]: v for k, v in pl_module_state_dict.items() if k.startswith(prefix)}


def get_text_encoder_state_dict_from_pretrained_mil_nce_checkpoint(
        checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    pl_module_state_dict = checkpoint["state_dict"]

    # Look for the corresponding encoder, with backward compatibility.
    prefix = "encoder.text_module." if any(k.startswith("encoder.text_module.") for k in pl_module_state_dict.keys()) \
        else "text_encoder."
    return {k[len(prefix):]: v for k, v in pl_module_state_dict.items() if k.startswith(prefix)}


class GlobalMaxPool1d(nn.Module):
    @overrides(check_signature=False)
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return t.max(dim=1)[0]


class MilNceTextEncoder(nn.Module):
    def __init__(self, output_size: int = 512, vocab_size: int = 66250, word_embedding_size: int = 300,
                 embedding: Optional[torch.Tensor] = None, hidden_size: int = 2048) -> None:
        super().__init__()
        # noinspection SpellCheckingInspection
        self.word_embd = nn.Embedding(vocab_size, word_embedding_size) if embedding is None \
            else nn.Embedding.from_pretrained(embedding)
        self.fc1 = nn.Linear(self.word_embd.embedding_dim, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = GlobalMaxPool1d()
        self.fc2 = nn.Linear(hidden_size, output_size)

    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        text = self.word_embd(input_ids)
        text = self.relu(self.fc1(text))
        text = self.max_pooling(text)
        return self.fc2(text)


def truncate_or_pad_1d_tensor(tensor: torch.Tensor, size: int, fill_value: Any = 0) -> torch.Tensor:
    if len(tensor) >= size:
        return tensor[:size]
    else:
        padded_tensor = torch.full((size,), fill_value, dtype=tensor.dtype, device=tensor.device,
                                   requires_grad=tensor.requires_grad)
        padded_tensor[:len(tensor)] = tensor
        return padded_tensor


class MilNceTokenizer:
    RE_WORD = re.compile(r"[\w']+")

    def __init__(self, vocab: Mapping[str, int], max_tokens: int = 20, lowercase: bool = True) -> None:
        super().__init__()
        self.vocab = vocab
        self.max_tokens = max_tokens
        self.lowercase = lowercase

        self.indices_to_tokens = {i: token for token, i in vocab.items()}

    def _tokenize(self, text: str) -> Iterator[str]:
        if self.lowercase:
            text = text.lower()
        return self.RE_WORD.findall(text)

    def _index(self, tokens: Iterable[str]) -> torch.Tensor:
        tokens_in_vocab_tensor = torch.tensor([self.vocab[word] for word in tokens if word in self.vocab],
                                              dtype=torch.long)
        return truncate_or_pad_1d_tensor(tokens_in_vocab_tensor, self.max_tokens)

    def __call__(self, text: str) -> TYPE_TEXT_INPUT:
        return {"input_ids": self._index(self._tokenize(text))}

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.indices_to_tokens[i] for i in ids if i != 0)


class MilNceVideoTextEncoder(VideoTextEncoder):
    def __init__(self, vocab_path: TYPE_PATH = "https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy",
                 pretrained_path: Optional[TYPE_PATH] = None, max_tokens: int = 20, num_frames: int = 16) -> None:
        super().__init__()
        self.video_encoder = S3DG()
        self.text_encoder = MilNceTextEncoder()

        vocab: Mapping[str, int] = {t.item(): i + 1 for i, t in enumerate(np.load(cached_path(vocab_path)))}
        self.tokenizer = MilNceTokenizer(vocab=vocab, max_tokens=max_tokens)

        self.num_frames = num_frames

        if pretrained_path:
            pretrained_path = cached_path(pretrained_path)
            self.video_encoder.load_state_dict(load_pretrained_video_encoder(pretrained_path,  # noqa
                                                                             map_location="cpu"))
            self.text_encoder.load_state_dict(load_pretrained_text_encoder(pretrained_path,  # noqa
                                                                             map_location="cpu"))

    @overrides(check_signature=False)
    def encode_video(self, video: TYPE_VIDEO_INPUT) -> torch.Tensor:
        return self.video_encoder(video)

    @overrides(check_signature=False)
    def encode_text(self, text: TYPE_TEXT_INPUT) -> torch.Tensor:
        return self.text_encoder(text["input_ids"])

    def _tokenize(self, texts: Iterable[str]) -> TYPE_TEXT_INPUT:
        tokenized = [self.tokenizer(text) for text in texts]
        return {k: torch.stack([t[k] for t in tokenized]) for k in next(iter(tokenized), [])}

    @overrides
    def get_tokenizer(self) -> TYPE_TOKENIZER:
        return self._tokenize

    @overrides
    def decode_text(self, text: TYPE_TEXT_INPUT) -> Iterator[str]:
        for text_instance in text["input_ids"]:
            yield self.tokenizer.decode(text_instance.tolist())

    @overrides
    def get_train_frame_sampler(self) -> FrameSampler:
        raise NotImplementedError

    @overrides
    def get_eval_frame_sampler(self) -> FrameSampler:
        return ConsecutiveFrameSampler(self.num_frames, fps=5)

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
