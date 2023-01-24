import functools
import glob
import os
from typing import Iterable, Literal, Mapping, Optional, Tuple

from cached_path import cached_path
from overrides import overrides
from torch.utils.data import DataLoader

from aligner.data.ucf import UCF_101_TEMPLATES
from aligner.data.video_data_module import VideoClassificationDataModule
from aligner.data.video_dataset import VideoDataset
from util.typing_utils import TYPE_PATH

TRAIN_TAG = 1
TEST_TAG = 2


class Hmdb(VideoDataset):
    def __init__(self, categories: Mapping[str, int], splits_folder: TYPE_PATH, split: Literal[1, 2, 3],
                 tag: Literal[1, 2], videos_folder: TYPE_PATH, **kwargs) -> None:
        self.categories = categories

        videos_folder = cached_path(videos_folder)

        video_paths = []
        for path in glob.iglob(os.path.join(cached_path(splits_folder), f"*_test_split{split}.txt")):
            category = os.path.basename(path).rsplit("_", maxsplit=2)[0]
            with open(path) as file:
                for line in file:
                    filename, file_tag = line.strip().split(maxsplit=1)
                    file_tag = int(file_tag)
                    if file_tag == tag:
                        video_paths.append(os.path.join(videos_folder, category, filename))
        super().__init__(video_paths=video_paths, **kwargs)

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
        category = folder_name.replace("_", " ")
        return category, self.categories[category]


class HmdbDataModule(VideoClassificationDataModule):  # noqa
    categories = {}  # Necessary because it's an abstract property. See https://stackoverflow.com/a/42529760/1165181

    def __init__(self, categories_file_path: TYPE_PATH, splits_folder: TYPE_PATH, split: Literal[1, 2, 3],
                 videos_folder: TYPE_PATH, **kwargs) -> None:
        super().__init__(**kwargs)
        self.splits_folder = splits_folder
        self.split = split
        self.videos_folder = videos_folder

        with open(cached_path(categories_file_path)) as file:
            self.categories = {line.strip(): i for i, line in enumerate(file)}

    @property
    @overrides
    def templates(self) -> Optional[Iterable[str]]:
        return UCF_101_TEMPLATES

    @overrides
    def train_dataloader(self) -> DataLoader:
        dataset = Hmdb(categories=self.categories, splits_folder=self.splits_folder, split=self.split,
                       tag=TRAIN_TAG, videos_folder=self.videos_folder,  # noqa
                       **self._create_dataset_encoder_kwargs(train=True))
        return self._create_dataloader(dataset, train=True)

    @overrides
    def val_dataloader(self) -> DataLoader:
        dataset = Hmdb(categories=self.categories, splits_folder=self.splits_folder, split=self.split,
                       tag=TEST_TAG, videos_folder=self.videos_folder,  # noqa
                       **self._create_dataset_encoder_kwargs(train=False))
        return self._create_dataloader(dataset, train=False)
