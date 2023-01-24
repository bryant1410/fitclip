import functools
import os
import re
from typing import Iterable, Mapping, Optional, Tuple

from cached_path import cached_path
from overrides import overrides
from torch.utils.data import DataLoader

from aligner.data.video_data_module import VideoClassificationDataModule
from aligner.data.video_dataset import VideoDataset
from util.typing_utils import TYPE_PATH

CATEGORIES_FILE_PATH = ("https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip!"
                        "ucfTrainTestlist/classInd.txt")
VAL_FILE_LIST_PATH = ("https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip!"
                      "ucfTrainTestlist/testlist01.txt")
VAL_VIDEOS_FOLDER = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar!UCF-101"

RE_CAPITALIZED_WORDS = re.compile(r"[a-zA-Z][^A-Z]*")

UCF_101_TEMPLATES = [  # From https://github.com/openai/CLIP/blob/main/data/prompts.md#ucf101
    "a photo of a person {}.",
    "a video of a person {}.",
    "a example of a person {}.",
    "a demonstration of a person {}.",
    "a photo of the person {}.",
    "a video of the person {}.",
    "a example of the person {}.",
    "a demonstration of the person {}.",
    "a photo of a person using {}.",
    "a video of a person using {}.",
    "a example of a person using {}.",
    "a demonstration of a person using {}.",
    "a photo of the person using {}.",
    "a video of the person using {}.",
    "a example of the person using {}.",
    "a demonstration of the person using {}.",
    "a photo of a person doing {}.",
    "a video of a person doing {}.",
    "a example of a person doing {}.",
    "a demonstration of a person doing {}.",
    "a photo of the person doing {}.",
    "a video of the person doing {}.",
    "a example of the person doing {}.",
    "a demonstration of the person doing {}.",
    "a photo of a person during {}.",
    "a video of a person during {}.",
    "a example of a person during {}.",
    "a demonstration of a person during {}.",
    "a photo of the person during {}.",
    "a video of the person during {}.",
    "a example of the person during {}.",
    "a demonstration of the person during {}.",
    "a photo of a person performing {}.",
    "a video of a person performing {}.",
    "a example of a person performing {}.",
    "a demonstration of a person performing {}.",
    "a photo of the person performing {}.",
    "a video of the person performing {}.",
    "a example of the person performing {}.",
    "a demonstration of the person performing {}.",
    "a photo of a person practicing {}.",
    "a video of a person practicing {}.",
    "a example of a person practicing {}.",
    "a demonstration of a person practicing {}.",
    "a photo of the person practicing {}.",
    "a video of the person practicing {}.",
    "a example of the person practicing {}.",
    "a demonstration of the person practicing {}.",
]


def _folder_name_to_category(folder_name: str) -> str:
    return " ".join(RE_CAPITALIZED_WORDS.findall(folder_name))


class Ucf(VideoDataset):
    def __init__(self, categories: Mapping[str, int], file_list_path: TYPE_PATH, videos_folder: TYPE_PATH,
                 **kwargs) -> None:
        self.categories = categories
        videos_folder = cached_path(videos_folder)
        with open(cached_path(file_list_path)) as file:
            video_ids = (stripped_line for line in file if (stripped_line := line.strip()))
            super().__init__(video_paths=(os.path.join(videos_folder, path) for path in video_ids), **kwargs)

    @functools.lru_cache
    @overrides
    def _get_video_id(self, video_idx: int) -> str:
        path = self.video_paths[video_idx]

        folder_path, filename = os.path.split(path)
        folder_name = os.path.basename(folder_path)
        return os.path.join(folder_name, filename)

    @functools.lru_cache
    @overrides
    def _get_target(self, video_idx: int) -> Tuple[str, int]:
        video_id = self._get_video_id(video_idx)
        folder_name = os.path.dirname(video_id)
        category = _folder_name_to_category(folder_name)
        return category, self.categories[category]


class UcfDataModule(VideoClassificationDataModule):  # noqa
    categories = {}  # Necessary because it's an abstract property. See https://stackoverflow.com/a/42529760/1165181

    def __init__(self, categories_file_path: TYPE_PATH = CATEGORIES_FILE_PATH,
                 val_file_list_path: TYPE_PATH = VAL_FILE_LIST_PATH,
                 val_videos_folder: TYPE_PATH = VAL_VIDEOS_FOLDER, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val_file_list_path = val_file_list_path
        self.val_videos_folder = val_videos_folder

        with open(cached_path(categories_file_path)) as file:
            self.categories = {}
            for line in file:
                id_, folder_name = line.strip().split()
                self.categories[_folder_name_to_category(folder_name)] = int(id_) - 1

    @property
    @overrides
    def templates(self) -> Optional[Iterable[str]]:
        return UCF_101_TEMPLATES

    @overrides
    def val_dataloader(self) -> DataLoader:
        dataset = Ucf(categories=self.categories, file_list_path=self.val_file_list_path,
                      videos_folder=self.val_videos_folder, **self._create_dataset_encoder_kwargs(train=False))
        return self._create_dataloader(dataset, train=False)
