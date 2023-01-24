import functools
import os

import pandas as pd
from cached_path import cached_path
from overrides import overrides
from torch.utils.data import DataLoader
from torchvision.datasets.folder import IMG_EXTENSIONS

from aligner.data.video_data_module import VideoTextDataModule
from aligner.data.video_dataset import VideoDataset
from aligner.data.video_text_dataset import VideoTextDataset
from util.typing_utils import TYPE_PATH
from util.video_utils import get_videos_in_folder


class ConceptualCaptions(VideoTextDataset):
    def __init__(self, video_info_file_path: TYPE_PATH, videos_folder: TYPE_PATH, **kwargs) -> None:
        self.video_info = pd.read_csv(cached_path(video_info_file_path), names=["name", "url", "video_id"],
                                      index_col="video_id")
        # The version of CC3M used here was downloaded by keeping the original filenames. The issue is that the
        # filenames repeat, and only one of the files was kept, but we don't know which one it is from the
        # information file with the captions. So as a workaround, we remove the duplicate video IDs:
        self.video_info = self.video_info[~self.video_info.index.duplicated(keep=False)]

        video_paths = sorted(path
                             for path in get_videos_in_folder(cached_path(videos_folder), extensions=IMG_EXTENSIONS)
                             if os.path.basename(path) in self.video_info.index)
        super().__init__(video_paths=video_paths, **kwargs)

    @functools.lru_cache
    @overrides
    def _get_video_id(self, video_idx: int) -> str:
        return os.path.basename(self.video_paths[video_idx])

    @overrides
    def _get_target(self, video_idx: int) -> str:
        video_id = self._get_video_id(video_idx)
        return self.video_info.loc[video_id, "name"]


class ConceptualCaptionsDataModule(VideoTextDataModule):  # noqa
    def __init__(self, train_video_info_file_path: TYPE_PATH, train_videos_folder: TYPE_PATH,
                 val_video_info_file_path: TYPE_PATH, val_videos_folder: TYPE_PATH, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train_video_info_file_path = train_video_info_file_path
        self.train_videos_folder = train_videos_folder
        self.val_video_info_file_path = val_video_info_file_path
        self.val_videos_folder = val_videos_folder

    def _dataset(self, video_info_file_path: TYPE_PATH, videos_folder: TYPE_PATH, train: bool) -> VideoDataset:
        return ConceptualCaptions(video_info_file_path=video_info_file_path, videos_folder=videos_folder,
                                  **self._create_dataset_encoder_kwargs(train=train))

    @overrides
    def train_dataloader(self) -> DataLoader:
        dataset = self._dataset(video_info_file_path=self.train_video_info_file_path,
                                videos_folder=self.train_videos_folder, train=True)
        return self._create_dataloader(dataset, train=True)

    @overrides
    def val_dataloader(self) -> DataLoader:
        dataset = self._dataset(video_info_file_path=self.val_video_info_file_path,
                                videos_folder=self.val_videos_folder, train=False)
        return self._create_dataloader(dataset, train=False)
