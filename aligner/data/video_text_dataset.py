from abc import ABC
from typing import Mapping, Union

from torch.utils.data.dataloader import default_collate

from aligner.data.tokenizer_collate import MappingTokenizerCollate
from aligner.data.video_dataset import VideoDataset
from aligner.encoder.video_text_encoder import TYPE_TOKENIZER


class VideoTextDataset(VideoDataset, ABC):
    def __init__(self, tokenizer: Union[TYPE_TOKENIZER, Mapping[str, TYPE_TOKENIZER]], target_key_name: str = "text",
                 **kwargs) -> None:
        super().__init__(target_key_name=target_key_name, **kwargs)
        self.collate = MappingTokenizerCollate(tokenizer, target_key_name,
                                               default_collate_fn=getattr(self, "collate", default_collate))
