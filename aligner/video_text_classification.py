import logging
import math
from typing import Any, Iterable, Mapping, Optional, Sequence, TypeVar

import torch
from overrides import overrides
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.utilities.apply_func import apply_to_collection
from torch import nn
from torchmetrics import Accuracy, Metric

from aligner.encoder.video_text_encoder import VideoTextEncoder
from aligner.metrics import MedianRank
from aligner.video_text_module import VideoTextLightningModule
from util import iter_utils

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


def batch_tokenized_text(tokenized: Mapping[str, Sequence[T]], n: int) -> Iterable[Mapping[str, T]]:
    tokenized_dicts = {k: iter(iter_utils.batch_sequence(v, n)) for k, v in tokenized.items()}
    length = math.ceil(len(next(iter(tokenized.values()))) / n)
    for _ in range(length):
        yield {k: next(tokenized_dicts[k]) for k in tokenized}


class VideoTextClassificationLightningModule(VideoTextLightningModule):  # noqa
    def __init__(self, encoder: VideoTextEncoder, labels: Iterable[str], templates: Optional[Iterable[str]],
                 return_metrics_by_class: bool = False, **kwargs) -> None:
        super().__init__(encoder, **kwargs)

        labels = list(labels)
        label_count = len(labels)

        # If different templates are provided, we used them for each label
        # and reset the labels to be {labels} x {templates}.

        if templates:
            templates = list(templates)

            self.template_count = len(templates)

            labels = [template.format(label) for label in labels for template in templates]
        else:
            self.template_count = 1

        # We tokenize all the labels but defer the encoding until the model is in the device.

        tokenized_labels = encoder.get_tokenizer()(labels)
        device = next(encoder.parameters()).device
        tokenized_labels = apply_to_collection(tokenized_labels, torch.Tensor, torch.Tensor.to, device)
        self.tokenized_labels = nn.ParameterDict(apply_to_collection(tokenized_labels, torch.Tensor, nn.Parameter,
                                                                     requires_grad=False))

        # We encode just one label to allocate the size correctly.
        encoded_text = self.encoder.encode_text({k: v[:1] for k, v in tokenized_labels.items()})
        self.encoded_labels = nn.Parameter(torch.empty(label_count, encoded_text.shape[-1]), requires_grad=False)

        self.metrics: Mapping[str, Metric] = nn.ModuleDict({"a1": Accuracy(), "a5": Accuracy(top_k=5),
                                                            "mr": MedianRank()})

        if return_metrics_by_class:
            self.metrics_by_class = nn.ModuleDict({f"a1_{k}": Accuracy() for k in range(label_count)})
        else:
            self.metrics_by_class = None

    def _on_start(self) -> None:
        # Note that for training they should be encoded at running time, not here.
        # But we aren't training any text classification model but evaluating them.
        #
        # We compute them here and not during init because here the model is already in the device.
        # This is especially much faster than in CPU (init) when using templates.
        batch_size = 32

        callback = next(callback for callback in self.trainer.callbacks if isinstance(callback, RichProgressBar))
        progress = callback.progress
        if self.trainer.is_global_zero:
            progress_task = progress.add_task(
                description="Encoding the labels",
                total=math.ceil(len(next(iter(self.tokenized_labels.values()))) / batch_size))
        else:
            progress_task = None

        encoded_label_list = []

        for tokenized_labels_batch in batch_tokenized_text(self.tokenized_labels, batch_size):
            encoded_label_list.append(self.encoder.encode_text(tokenized_labels_batch))

            if progress_task is not None:
                progress.update(progress_task, advance=1)

        encoded_labels = torch.cat(encoded_label_list)
        encoded_labels = encoded_labels.reshape(-1, self.template_count, encoded_labels.shape[1]).mean(dim=1)
        self.encoded_labels.copy_(encoded_labels)

        if progress_task is not None:
            # If we remove it, it later fails, not sure why. So we just hide it.
            progress.update(progress_task, visible=False)

    @overrides
    def on_validation_start(self) -> None:
        self._on_start()

    @overrides
    def on_test_start(self) -> None:
        self._on_start()

    @overrides
    def on_predict_start(self) -> None:
        self._on_start()

    @overrides(check_signature=False)
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_video(video) @ self.encoded_labels.T

    @overrides(check_signature=False)
    def validation_step(self, batch: Mapping[str, Any], _batch_idx: int = 0) -> None:
        scores = self(batch["video"])
        label_id = batch["target"][1]
        for name, metric in self.metrics.items():
            metric(scores, label_id)
            # Note that we need to pass the batch size in the first step log
            # as it can't be easily inferred by PL in our case.
            self.log(name, metric, prog_bar=True, batch_size=len(batch["video"]))

        if self.metrics_by_class is not None:
            for scores_instance, label_id_instance in zip(scores, label_id):
                metric = self.metrics_by_class[f"a1_{label_id_instance}"]
                metric(scores_instance.unsqueeze(0), label_id_instance.unsqueeze(0))
                self.log(f"a1_{label_id_instance}", metric, batch_size=1)

    @overrides(check_signature=False)
    def predict_step(self, batch: Mapping[str, Any], _batch_idx: int = 0) -> Mapping[str, torch.Tensor]:
        return {
            "predictions": self(batch["video"]).argmax(dim=-1),
            "labels": batch["target"][1],
            "video_ids": batch["video_id"],
        }
