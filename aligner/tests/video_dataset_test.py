import decord
import numpy as np
from cached_path import cached_path

from aligner.data.video_dataset import time_to_indices


def test_seek() -> None:
    # noinspection SpellCheckingInspection
    video_reader = decord.VideoReader(cached_path("https://mdn.github.io/learning-area/html/multimedia-and-embedding/"
                                                  "video-and-audio-content/rabbit320.webm"))
    assert time_to_indices(video_reader, 2.5) == 75


def test_seek_array() -> None:
    # noinspection SpellCheckingInspection
    video_reader = decord.VideoReader(cached_path("https://mdn.github.io/learning-area/html/multimedia-and-embedding/"
                                                  "video-and-audio-content/rabbit320.webm"))
    assert (time_to_indices(video_reader, [2.5, 4.2]) == np.array([75, 126])).all()
