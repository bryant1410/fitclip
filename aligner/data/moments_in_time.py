import functools
import os
from typing import Mapping, Tuple

import pandas as pd
from cached_path import cached_path
from overrides import overrides
from torch.utils.data import DataLoader

from aligner.data.video_data_module import VideoClassificationDataModule
from aligner.data.video_dataset import VideoDataset
from util.typing_utils import TYPE_PATH
from util.video_utils import get_sorted_videos_in_folder

CATEGORIES_FILE_PATH = "/datasets/moments-in-time/moments_categories.txt"
VAL_VIDEO_INFO_FILE_PATH = "/datasets/moments-in-time/validationSet.csv"
VAL_VIDEOS_FOLDER = "/datasets/moments-in-time/validation"


class MomentsInTime(VideoDataset):
    def __init__(self, categories: Mapping[str, int], video_info_file_path: TYPE_PATH, videos_folder: TYPE_PATH,
                 **kwargs) -> None:
        super().__init__(video_paths=get_sorted_videos_in_folder(cached_path(videos_folder)), **kwargs)
        self.categories = categories
        self.video_info = pd.read_csv(cached_path(video_info_file_path), names=["path", "category", "agreement",
                                                                                "disagreement"], index_col="path")

    @functools.lru_cache
    @overrides
    def _get_video_id(self, video_idx: int) -> str:
        path = self.video_paths[video_idx]

        folder_path, filename = os.path.split(path)
        folder_name = os.path.basename(folder_path)
        return os.path.join(folder_name, filename)

    @overrides
    def _get_target(self, video_idx: int) -> Tuple[str, int]:
        video_id = self._get_video_id(video_idx)
        category = self.video_info.loc[video_id, "category"]
        return category, self.categories[category]


class MomentsInTimeDataModule(VideoClassificationDataModule):  # noqa
    categories = {}  # Necessary because it's an abstract property. See https://stackoverflow.com/a/42529760/1165181

    def __init__(self, categories_file_path: TYPE_PATH = CATEGORIES_FILE_PATH,
                 val_video_info_file_path: TYPE_PATH = VAL_VIDEO_INFO_FILE_PATH,
                 val_videos_folder: TYPE_PATH = VAL_VIDEOS_FOLDER, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val_video_info_file_path = val_video_info_file_path
        self.val_videos_folder = val_videos_folder

        with open(cached_path(categories_file_path)) as file:
            self.categories = {}
            for line in file:
                category, id_ = line.rstrip().split(",")
                self.categories[category] = int(id_)

    @overrides
    def val_dataloader(self) -> DataLoader:
        dataset = MomentsInTime(categories=self.categories, video_info_file_path=self.val_video_info_file_path,
                                videos_folder=self.val_videos_folder,
                                **self._create_dataset_encoder_kwargs(train=False))
        return self._create_dataloader(dataset, train=False)
