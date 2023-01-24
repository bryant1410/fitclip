import json
import os
import random
from typing import Literal

import pandas as pd
from cached_path import cached_path
from overrides import overrides
from torch.utils.data import DataLoader

from aligner.data.video_data_module import VideoTextDataModule
from aligner.data.video_dataset import VideoDataset
from aligner.data.video_text_dataset import VideoTextDataset
from util.typing_utils import TYPE_PATH
from util.video_utils import get_sorted_videos_in_folder

TYPE_CAPTION_SAMPLING_STRATEGY = Literal["first", "random"]


class MsrVtt(VideoTextDataset):
    def __init__(self, videos_folder: TYPE_PATH, file_list_path: TYPE_PATH, annotations_path: TYPE_PATH,
                 caption_sampling_strategy: TYPE_CAPTION_SAMPLING_STRATEGY, **kwargs) -> None:
        with open(cached_path(file_list_path)) as file:
            video_ids = {stripped_line for line in file if (stripped_line := line.strip())}  # noqa

        video_paths = (path
                       for path in get_sorted_videos_in_folder(cached_path(videos_folder))
                       if os.path.basename(path).split(".", maxsplit=1)[0] in video_ids)

        super().__init__(video_paths=video_paths, **kwargs)

        self.caption_sampling_strategy = caption_sampling_strategy

        with open(cached_path(annotations_path)) as file:
            metadata = json.load(file)

        self.video_info = pd.DataFrame(metadata["annotations"])
        self.video_info.set_index("image_id", inplace=True)

    @overrides
    def _get_target(self, video_idx: int) -> str:
        video_id = self._get_video_id(video_idx)
        captions = self.video_info.loc[video_id, "caption"]
        if self.caption_sampling_strategy == "first":
            return captions[0]
        elif self.caption_sampling_strategy == "random":
            return random.choice(captions)
        else:
            raise ValueError(f"Invalid choice of caption sampling strategy: {self.caption_sampling_strategy}")


class MsrVttDataModule(VideoTextDataModule):  # noqa
    def __init__(self,
                 base_path: TYPE_PATH = "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip!MSRVTT/",
                 train_file_list_rel_path: TYPE_PATH = "train_list_jsfusion.txt",  # 1K-A split
                 val_file_list_rel_path: TYPE_PATH = "val_list_jsfusion.txt", **kwargs) -> None:
        super().__init__(**kwargs)
        base_path = cached_path(base_path)
        self.videos_folder = os.path.join(base_path, "videos/all")
        self.annotation_path = os.path.join(base_path, "annotation/MSR_VTT.json")
        self.train_file_list_path = os.path.join(base_path, "structured-symlinks", train_file_list_rel_path)
        self.val_file_list_path = os.path.join(base_path, "structured-symlinks", val_file_list_rel_path)

    def _dataset(self, file_list_path: TYPE_PATH, caption_sampling_strategy: TYPE_CAPTION_SAMPLING_STRATEGY,
                 train: bool) -> VideoDataset:
        return MsrVtt(videos_folder=self.videos_folder, file_list_path=file_list_path,
                      annotations_path=self.annotation_path, caption_sampling_strategy=caption_sampling_strategy,
                      **self._create_dataset_encoder_kwargs(train=train))

    @overrides
    def train_dataloader(self) -> DataLoader:
        dataset = self._dataset(file_list_path=self.train_file_list_path, caption_sampling_strategy="random",
                                train=True)
        return self._create_dataloader(dataset, train=True)

    @overrides
    def val_dataloader(self) -> DataLoader:
        dataset = self._dataset(file_list_path=self.val_file_list_path, caption_sampling_strategy="first", train=False)
        return self._create_dataloader(dataset, train=False)
