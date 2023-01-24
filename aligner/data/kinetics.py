import os
from typing import Iterable, Mapping, Optional, Tuple

import pandas as pd
from cached_path import cached_path
from overrides import overrides
from torch.utils.data import DataLoader

from aligner.data.video_data_module import VideoClassificationDataModule
from aligner.data.video_dataset import VideoDataset
from util.typing_utils import TYPE_PATH
from util.video_utils import get_sorted_videos_in_folder


class Kinetics(VideoDataset):
    def __init__(self, categories: Mapping[str, int], video_info_file_path: TYPE_PATH, videos_folder: TYPE_PATH,
                 filter_videos_from_info_file: bool = False, **kwargs) -> None:
        self.categories = categories

        self.video_info = pd.read_csv(cached_path(video_info_file_path))
        self.video_info["video_id"] = \
            self.video_info.agg(lambda row: f"{row.youtube_id}_{row.time_start:06}_{row.time_end:06}", axis=1)
        self.video_info.set_index("video_id", inplace=True)

        if filter_videos_from_info_file:
            video_paths = (cached_path(os.path.join(videos_folder, f"{video_id}.mp4"))
                           for video_id, _ in self.video_info.iterrows())
        else:
            video_paths = get_sorted_videos_in_folder(cached_path(videos_folder))

        super().__init__(video_paths=video_paths, **kwargs)

    @overrides
    def _get_target(self, video_idx: int) -> Tuple[str, int]:
        video_id = self._get_video_id(video_idx)
        category = self.video_info.loc[video_id, "label"]
        return category, self.categories[category]


class KineticsDataModule(VideoClassificationDataModule):  # noqa
    categories = {}  # Necessary because it's an abstract property. See https://stackoverflow.com/a/42529760/1165181

    def __init__(self, categories_file_path: TYPE_PATH, train_video_info_file_path: TYPE_PATH,
                 train_videos_folder: TYPE_PATH, val_video_info_file_path: TYPE_PATH, val_videos_folder: TYPE_PATH,
                 test_video_info_file_path: TYPE_PATH, test_videos_folder: TYPE_PATH,
                 train_filter_videos_from_info_file: bool = False, val_filter_videos_from_info_file: bool = False,
                 test_filter_videos_from_info_file: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train_video_info_file_path = train_video_info_file_path
        self.train_videos_folder = train_videos_folder
        self.train_filter_videos_from_info_file = train_filter_videos_from_info_file
        self.val_video_info_file_path = val_video_info_file_path
        self.val_videos_folder = val_videos_folder
        self.val_filter_videos_from_info_file = val_filter_videos_from_info_file
        self.test_video_info_file_path = test_video_info_file_path
        self.test_videos_folder = test_videos_folder
        self.test_filter_videos_from_info_file = test_filter_videos_from_info_file

        with open(cached_path(categories_file_path)) as file:
            self.categories = {line.strip(): i for i, line in enumerate(file)}

    @property
    @overrides
    def templates(self) -> Optional[Iterable[str]]:
        return [  # From https://github.com/openai/CLIP/blob/main/data/prompts.md#kinetics700
            "a photo of {}.",
            "a photo of a person {}.",
            "a photo of a person using {}.",
            "a photo of a person doing {}.",
            "a photo of a person during {}.",
            "a photo of a person performing {}.",
            "a photo of a person practicing {}.",
            "a video of {}.",
            "a video of a person {}.",
            "a video of a person using {}.",
            "a video of a person doing {}.",
            "a video of a person during {}.",
            "a video of a person performing {}.",
            "a video of a person practicing {}.",
            "a example of {}.",
            "a example of a person {}.",
            "a example of a person using {}.",
            "a example of a person doing {}.",
            "a example of a person during {}.",
            "a example of a person performing {}.",
            "a example of a person practicing {}.",
            "a demonstration of {}.",
            "a demonstration of a person {}.",
            "a demonstration of a person using {}.",
            "a demonstration of a person doing {}.",
            "a demonstration of a person during {}.",
            "a demonstration of a person performing {}.",
            "a demonstration of a person practicing {}.",
        ]

    def _dataset(self, video_info_file_path: TYPE_PATH, videos_folder: TYPE_PATH,
                 filter_videos_from_info_file: bool, train: bool) -> VideoDataset:
        return Kinetics(self.categories, video_info_file_path=video_info_file_path, videos_folder=videos_folder,
                        filter_videos_from_info_file=filter_videos_from_info_file,
                        **self._create_dataset_encoder_kwargs(train=train))

    @overrides
    def train_dataloader(self) -> DataLoader:
        dataset = self._dataset(video_info_file_path=self.train_video_info_file_path,
                                videos_folder=self.train_videos_folder,
                                filter_videos_from_info_file=self.train_filter_videos_from_info_file, train=True)
        return self._create_dataloader(dataset, train=True)

    @overrides
    def val_dataloader(self) -> DataLoader:
        dataset = self._dataset(video_info_file_path=self.val_video_info_file_path,
                                videos_folder=self.val_videos_folder,
                                filter_videos_from_info_file=self.val_filter_videos_from_info_file, train=False)
        return self._create_dataloader(dataset, train=False)

    @overrides
    def test_dataloader(self) -> DataLoader:
        dataset = self._dataset(video_info_file_path=self.test_video_info_file_path,
                                videos_folder=self.test_videos_folder,
                                filter_videos_from_info_file=self.test_filter_videos_from_info_file, train=False)
        return self._create_dataloader(dataset, train=False)
