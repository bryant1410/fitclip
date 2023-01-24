import json
import os
from collections import defaultdict

from cached_path import CACHE_DIR, _find_latest_cached, cached_path
from overrides import overrides
from torch.utils.data import DataLoader

from aligner.data.video_data_module import VideoTextDataModule
from aligner.data.video_text_dataset import VideoTextDataset
from util.typing_utils import TYPE_PATH

HASH_LIST_PATH = "https://raw.githubusercontent.com/LisaAnne/LocalizingMoments/master/data/yfcc100m_hash.txt"
VAL_ANNOTATION_PATH = "https://raw.githubusercontent.com/LisaAnne/LocalizingMoments/master/data/val_data.json"
VIDEOS_FOLDER = "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/"


class Didemo(VideoTextDataset):
    def __init__(self, videos_folder: TYPE_PATH, hash_list_path: TYPE_PATH, annotations_path: TYPE_PATH,
                 **kwargs) -> None:
        with open(cached_path(annotations_path)) as file:
            description_list_by_id = defaultdict(list)
            for video in json.load(file):
                description_list_by_id[video["video"]].append(video["description"])

        self.description_paragraph_by_id = {video_id: " ".join(descriptions)
                                            for video_id, descriptions in description_list_by_id.items()}

        with open(cached_path(hash_list_path)) as file:
            hash_by_flickr_id = {}
            for line in file:
                flickr_id, hash_ = line.strip().split("\t")
                hash_by_flickr_id[flickr_id] = hash_

        self.video_ids_by_path = {}
        for video_id in self.description_paragraph_by_id:
            flickr_id = video_id.split("_")[1]
            hash_ = hash_by_flickr_id[flickr_id]
            video_path_or_url = os.path.join(videos_folder, hash_[:3], hash_[3:6], f"{hash_}.mp4")
            # We only download some videos and not the whole folder.
            # But if it's already cached, we avoid sending a HEAD request. This is an issue if the file is updated,
            # but we assume it won't happen.
            video_path = _find_latest_cached(video_path_or_url, CACHE_DIR) or cached_path(video_path_or_url)
            self.video_ids_by_path[video_path] = video_id

        super().__init__(video_paths=self.video_ids_by_path.keys(), **kwargs)

    @overrides
    def _get_target(self, video_idx: int) -> str:
        video_path = self.video_paths[video_idx]
        video_id = self.video_ids_by_path[video_path]
        return self.description_paragraph_by_id[video_id]


class DidemoDataModule(VideoTextDataModule):  # noqa
    def __init__(self, videos_folder: TYPE_PATH = VIDEOS_FOLDER, hash_list_path: TYPE_PATH = HASH_LIST_PATH,
                 val_annotation_path: TYPE_PATH = VAL_ANNOTATION_PATH, **kwargs) -> None:
        super().__init__(**kwargs)
        self.videos_folder = videos_folder
        self.hash_list_path = hash_list_path
        self.val_annotation_path = val_annotation_path

    @overrides
    def val_dataloader(self) -> DataLoader:
        dataset = Didemo(videos_folder=self.videos_folder, hash_list_path=self.hash_list_path,
                         annotations_path=self.val_annotation_path, **self._create_dataset_encoder_kwargs(train=False))
        return self._create_dataloader(dataset, train=False)
