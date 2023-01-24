import os

import pandas as pd
from cached_path import cached_path
from overrides import overrides
from torch.utils.data import DataLoader

from aligner.data.video_data_module import VideoTextDataModule
from aligner.data.video_dataset import VideoDataset
from aligner.data.video_text_dataset import VideoTextDataset
from util.typing_utils import TYPE_PATH
from util.video_utils import get_sorted_videos_in_folder

TRAIN_VIDEO_INFO_FILE_PATH = "/datasets/webvid/results_2M_train.csv"
# noinspection SpellCheckingInspection
TRAIN_VIDEOS_FOLDER = "/datasets/webvid/videos_low_resolution/train/webvid_lowres/"

VAL_VIDEO_INFO_FILE_PATH = "/datasets/webvid/results_2M_val.csv"
# noinspection SpellCheckingInspection
VAL_VIDEOS_FOLDER = "/datasets/webvid/videos_low_resolution/val/val_lowres/"


class WebVid(VideoTextDataset):
    def __init__(self, video_info_file_path: TYPE_PATH, videos_folder: TYPE_PATH,
                 filter_videos_from_info_file: bool = False, **kwargs) -> None:
        # noinspection SpellCheckingInspection
        self.video_info = pd.read_csv(cached_path(video_info_file_path), index_col="videoid", dtype={"videoid": str})

        if filter_videos_from_info_file:
            video_paths = (cached_path(os.path.join(videos_folder, f"{video_id}.mp4"))
                           for video_id, _ in self.video_info.iterrows())
        else:
            video_paths = get_sorted_videos_in_folder(cached_path(videos_folder))

        super().__init__(video_paths=video_paths, **kwargs)

    @overrides
    def _get_target(self, video_idx: int) -> str:
        video_id = self._get_video_id(video_idx)
        return self.video_info.loc[video_id, "name"]


class WebVidDataModule(VideoTextDataModule):  # noqa
    def __init__(self, train_video_info_file_path: TYPE_PATH = TRAIN_VIDEO_INFO_FILE_PATH,
                 train_videos_folder: TYPE_PATH = TRAIN_VIDEOS_FOLDER, train_filter_videos_from_info_file: bool = False,
                 val_video_info_file_path: TYPE_PATH = VAL_VIDEO_INFO_FILE_PATH,
                 val_videos_folder: TYPE_PATH = VAL_VIDEOS_FOLDER, val_filter_videos_from_info_file: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.train_video_info_file_path = train_video_info_file_path
        self.train_videos_folder = train_videos_folder
        self.train_filter_videos_from_info_file = train_filter_videos_from_info_file
        self.val_video_info_file_path = val_video_info_file_path
        self.val_videos_folder = val_videos_folder
        self.val_filter_videos_from_info_file = val_filter_videos_from_info_file

    def _dataset(self, video_info_file_path: TYPE_PATH, videos_folder: TYPE_PATH,
                 filter_videos_from_info_file: bool, train: bool) -> VideoDataset:
        return WebVid(video_info_file_path=video_info_file_path, videos_folder=videos_folder,
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
