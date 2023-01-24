import multiprocessing
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Union

import pytorch_lightning as pl
import torch.cuda
from overrides import overrides
from pytorch_lightning.utilities.apply_func import apply_to_collection
from torch.utils.data import DataLoader

from aligner.data.frame_sampler import FrameSampler
from aligner.data.video_dataset import VideoDataset
from aligner.encoder.video_encoder import TYPE_TRANSFORM, VideoEncoder
from aligner.encoder.video_text_encoder import VideoTextEncoder

ENCODER_OR_ENCODER_MAP = Union[VideoEncoder, Mapping[str, VideoEncoder]]


def precision_to_dtype(precision: Union[str, int]) -> torch.dtype:
    if precision == 32:
        return torch.float
    elif precision == 64:
        return torch.float64
    elif precision in {16, "mixed"}:
        return torch.float16
    else:
        raise ValueError(f"Unsupported precision value: {precision}")


class VideoDataModule(pl.LightningDataModule, ABC):
    def __init__(self, encoder: ENCODER_OR_ENCODER_MAP, batch_size: Optional[int] = 1,
                 eval_batch_size: Optional[int] = 32,
                 num_workers: int = multiprocessing.cpu_count() // max(torch.cuda.device_count(), 1)) -> None:
        super().__init__()
        self.encoder = encoder
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

    def _create_transform(self, train: bool) -> Union[TYPE_TRANSFORM, Mapping[str, TYPE_TRANSFORM]]:
        float_precision = self.trainer.precision_plugin.precision
        dtype = precision_to_dtype(float_precision)
        return apply_to_collection(self.encoder, VideoEncoder,
                                   lambda e: (e.get_train_transform if train else e.get_eval_transform)(dtype))

    def _create_frame_sampler(self, train: bool) -> Union[FrameSampler, Mapping[str, FrameSampler]]:
        return apply_to_collection(self.encoder, VideoEncoder,
                                   lambda e: e.get_train_frame_sampler() if train else e.get_eval_frame_sampler())

    def _create_dataset_encoder_kwargs(self, train: bool) -> MutableMapping[str, Any]:
        # FIXME: Disable the cache because it seems like a new dataset is created by PL every time.
        return {"frame_sampler": self._create_frame_sampler(train=train),
                "transform": self._create_transform(train=train),
                "pad_batch": apply_to_collection(self.encoder, VideoEncoder, lambda e: e.should_pad_batch),
                "cache": False}

    def _create_dataloader(self, dataset: VideoDataset, train: bool) -> DataLoader:
        # Drop last in train so the NCE loss isn't smaller in the charts for the last batch.
        # Also, don't waste one step with fewer memory, where we could have the next one with more memory.
        batch_size = self.batch_size if train else self.eval_batch_size
        return DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0, collate_fn=getattr(dataset, "collate", None),
                          shuffle=train, drop_last=train)

    @overrides
    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()


class VideoTextDataModule(VideoDataModule, ABC):
    def __init__(self, encoder: Union[VideoTextEncoder, Mapping[str, VideoTextEncoder]], **kwargs) -> None:
        super().__init__(encoder=encoder, **kwargs)

    @overrides
    def _create_dataset_encoder_kwargs(self, train: bool) -> MutableMapping[str, Any]:
        kwargs = super()._create_dataset_encoder_kwargs(train=train)
        kwargs["tokenizer"] = apply_to_collection(self.encoder, VideoEncoder, lambda e: e.get_tokenizer())
        return kwargs


class VideoClassificationDataModule(VideoDataModule, ABC):
    @property
    @abstractmethod
    def categories(self) -> Mapping[str, int]:
        raise NotImplementedError

    @property
    def templates(self) -> Optional[Iterable[str]]:
        return None
