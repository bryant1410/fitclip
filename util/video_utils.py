import os
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

from torchvision.datasets.video_utils import VideoClips

from util.typing_utils import TYPE_PATH

# From https://en.wikipedia.org/wiki/Video_file_format
VIDEO_FILE_EXTENSIONS = (".3g2", ".3gp", ".amv", ".asf", ".avi", ".drc", ".f4a", ".f4b", ".f4p", ".f4v", ".flv",
                         ".gif", ".gifv", ".m2ts", ".m2v", ".m4p", ".m4v", ".mkv", ".mng", ".mov", ".mp2", ".mp4",
                         ".mpe", ".mpeg", ".mpg", ".mpv", ".mts", ".mxf", ".nsv", ".ogg", ".ogv", ".qt", ".rm",
                         ".rmvb", ".roq", ".svi", ".ts", ".viv", ".vob", ".webm", ".wmv", ".yuv")


def get_videos_in_folder(path: TYPE_PATH,
                         extensions: Optional[Iterable[str]] = VIDEO_FILE_EXTENSIONS) -> Iterator[str]:
    extensions = None if extensions is None else tuple(extensions)
    for folder, _, filenames in os.walk(path, followlinks=True):
        for filename in filenames:
            if os.path.isfile(full_path := os.path.join(folder, filename)) \
                    and (not extensions or filename.lower().endswith(extensions)):
                yield full_path


def get_sorted_videos_in_folder(path: TYPE_PATH,
                                extensions: Optional[Iterable[str]] = VIDEO_FILE_EXTENSIONS,
                                key: Optional[Callable[[str], Any]] = None, reverse: bool = False) -> Iterator[str]:
    """Returns a sorted version of `get_videos_in_folder`.

    Even though this can be simply applied by the caller, the fact that the main use case of `get_videos_in_folder`
    is from a video dataset and that its order should be deterministic (but that `get_videos_in_folder` doesn't
    guarantee it) makes this function handy and a wake-up call for this issue.

    The videos in a PyTorch `Dataset` need to be deterministic e.g. for a distributed setting, when e.g. using
    `DistributedSampler` for it to guarantee each data sample is used once and only once between all processes.
    """
    return sorted(get_videos_in_folder(path, extensions), key=key, reverse=reverse)


def resample(num_frames: int, original_fps: float, new_fps: float) -> Sequence[int]:
    """Returns essentially the same as `VideoClips._resample_video_idx`. Unlike it, it always checks for the max frames
    (the mentioned function doesn't do it when it returns a `slice`)."""
    indices = VideoClips._resample_video_idx(num_frames, original_fps, new_fps)

    if isinstance(indices, slice) and indices.stop is None:
        indices = range(*indices.indices((indices.start or 0) + num_frames * indices.step))

    return indices
