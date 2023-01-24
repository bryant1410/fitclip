import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots_adjust
from torchvision.transforms.functional import to_pil_image

from aligner.encoder.video_text_encoder import VideoTextEncoder


def visualize_images_tensor(images: torch.Tensor) -> plt.Axes:
    """`images` has shape (N, C, H, W)."""
    grid = torchvision.utils.make_grid(images)

    fig, ax = plt.subplots()

    fig.tight_layout()
    subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    ax.autoscale_view("tight")

    ax.imshow(np.asarray(to_pil_image(grid)))

    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return ax


def debug_batch(video: torch.Tensor, text: torch.Tensor, encoder: VideoTextEncoder) -> None:
    video, text = video.detach().cpu(), text.detach().cpu()

    video = encoder.to_bchw(video)
    denormalized_images = encoder.denormalize_video_tensor(video).reshape(-1, *video.shape[2:])
    visualize_images_tensor(denormalized_images)
    plt.show()

    for decoded in encoder.decode_text(text):
        print(decoded)
