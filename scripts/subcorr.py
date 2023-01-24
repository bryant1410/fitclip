#!/usr/bin/env python
import argparse
import sys
from typing import Any, Callable, Iterable, MutableMapping, Optional, Sequence, Union

import PIL.Image
import clip
import decord
import numpy as np
import seaborn as sns
import torch
from clip.model import CLIP
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from spacy.tokens import Doc, Span


def get_video_info(path: str) -> MutableMapping[str, Any]:
    video_reader = decord.VideoReader(path)

    frame_indices = list(range(0, len(video_reader), 10))
    frames = [PIL.Image.fromarray(f) for f in video_reader.get_batch(frame_indices).asnumpy()]

    thumbnails_frame_indices = video_reader.get_key_indices()
    thumbnails = [PIL.Image.fromarray(f) for f in video_reader.get_batch(thumbnails_frame_indices).asnumpy()]

    thumbnails = [f.copy() for f in thumbnails]
    for thumbnail in thumbnails:
        thumbnail.thumbnail((64, 64))

    return {
        "frames": frames,
        "frame_times": video_reader.get_frame_timestamp(frame_indices).mean(axis=-1),  # noqa
        "thumbnails": thumbnails,
        "thumbnail_times": video_reader.get_frame_timestamp(thumbnails_frame_indices).mean(axis=-1),  # noqa
    }


def encode_visual(images: Iterable[PIL.Image.Image], clip_model: CLIP,
                  image_preprocessor: Callable[[PIL.Image.Image], torch.Tensor],
                  device: Optional[Any] = None) -> torch.Tensor:
    images = torch.stack([image_preprocessor(image) for image in images])

    if device is not None:
        images = images.to(device)

    with torch.inference_mode():
        encoded_images = clip_model.encode_image(images)
        return encoded_images / encoded_images.norm(dim=-1, keepdim=True)


def encode_text(text: str, clip_model: CLIP, device: Optional[Any] = None) -> torch.Tensor:
    tokenized_texts = clip.tokenize([text])

    if device is not None:
        tokenized_texts = tokenized_texts.to(device)

    with torch.inference_mode():
        encoded_texts = clip_model.encode_text(tokenized_texts)
        return encoded_texts / encoded_texts.norm(dim=-1, keepdim=True)


def text_probs(encoded_images: torch.Tensor, encoded_texts: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        # clip_model.logit_scale.exp() == 100
        return (100 * encoded_images @ encoded_texts.T).softmax(dim=0).squeeze(-1).cpu().numpy()  # noqa


def create_figure(times: Sequence[float], probs: Sequence[float], thumbnail_times: Sequence[float],
                  thumbnails: Iterable[PIL.Image.Image], title: Union[Doc, Span, str]) -> plt.Axes:
    # noinspection SpellCheckingInspection
    sns.set(rc={"figure.figsize": (1.0 * len(thumbnail_times), 1.5)})

    ax = sns.lineplot(x=times, y=probs)

    plt.xticks(thumbnail_times)

    ax.set_title(title.text if isinstance(title, (Doc, Span)) else title, fontsize=35, y=0.6)
    ax.set(xlabel="time", ylabel="probability")

    plt.fill_between(times, probs)

    if isinstance(title, (Doc, Span)):
        start_time = title[0]._.start_time
        end_time = title[-1]._.end_time

        plt.axvspan(start_time, end_time, alpha=0.5, color="red")

    for i, (time, thumbnail) in enumerate(zip(thumbnail_times, thumbnails)):
        im = OffsetImage(thumbnail, axes=ax)
        ab = AnnotationBbox(im, (time, 0), xybox=(0, -60), frameon=False, boxcoords="offset points", pad=0)
        ax.add_artist(ab)

    plt.margins(x=0, tight=True)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    return ax


def create_figure_for_text(encoded_frames: torch.Tensor, text: Union[Doc, Span, str], clip_model: CLIP,
                           times: Sequence[float], thumbnail_times: Sequence[float],
                           thumbnails: Iterable[PIL.Image.Image]) -> plt.Axes:
    encoded_texts = encode_text(text.text if isinstance(text, (Doc, Span)) else text, clip_model,
                                device=encoded_frames.device)
    probs = text_probs(encoded_frames, encoded_texts)
    return create_figure(times, probs, thumbnail_times, thumbnails, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", metavar="PATH")
    return parser.parse_args()


def main() -> None:
    sns.set_theme()

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, image_preprocessor = clip.load("ViT-B/16", device=device)

    # noinspection SpellCheckingInspection
    video_info = get_video_info(args.path)

    encoded_frames = encode_visual(video_info["frames"], clip_model, image_preprocessor, device=device)

    for text in sys.stdin:
        if text := text.strip():
            create_figure_for_text(encoded_frames, text, clip_model, video_info["frame_times"],
                                   video_info["thumbnail_times"], video_info["thumbnails"])
            plt.show()


if __name__ == "__main__":
    main()
