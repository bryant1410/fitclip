import os
from glob import iglob
from typing import Optional, Tuple

import pandas as pd
from cached_path import cached_path
from overrides import overrides
from torch.utils.data import DataLoader

from aligner.data.video_data_module import VideoTextDataModule
from aligner.data.video_text_dataset import VideoTextDataset
from util.typing_utils import TYPE_PATH

VAL_VIDEO_INFO_FILE_PATH = "https://raw.githubusercontent.com/antoine77340/MIL-NCE_HowTo100M/master/csv/" \
                           "validation_youcook.csv"
# Videos can also be obtained from https://www.rocq.inria.fr/cluster-willow/amiech/Youcook2_val.zip!validation
VAL_VIDEOS_FOLDER = "/datasets/yc2_mil_nce_val/"


class YouCook2(VideoTextDataset):
    def __init__(self, video_info_file_path: TYPE_PATH, videos_folder: TYPE_PATH, **kwargs) -> None:
        self.video_info = pd.read_csv(cached_path(video_info_file_path), dtype={"task": str})

        video_folder = cached_path(videos_folder)
        video_paths = (next(iglob(os.path.join(video_folder, row.task, f"{row.video_id}.*")))
                       for _, row in self.video_info.iterrows())

        super().__init__(video_paths=video_paths, **kwargs)

    @overrides
    def _get_target(self, video_idx: int) -> str:
        return self.video_info.loc[video_idx, "text"]

    @overrides
    def _get_times(self, video_idx: int) -> Tuple[Optional[float], Optional[float]]:
        row = self.video_info.loc[video_idx]
        return row.start, row.end


class YouCook2DataModule(VideoTextDataModule):  # noqa
    def __init__(self, val_video_info_file_path: TYPE_PATH = VAL_VIDEO_INFO_FILE_PATH,
                 val_videos_folder: TYPE_PATH = VAL_VIDEOS_FOLDER, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val_video_info_file_path = val_video_info_file_path
        self.val_videos_folder = val_videos_folder

    @overrides
    def val_dataloader(self) -> DataLoader:
        dataset = YouCook2(video_info_file_path=self.val_video_info_file_path, videos_folder=self.val_videos_folder,
                           **self._create_dataset_encoder_kwargs(train=False))
        return self._create_dataloader(dataset, train=False)
