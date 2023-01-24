from collections import OrderedDict
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.distributed.nn
from overrides import overrides
from torch import nn
from torchmetrics import Metric, Recall

from aligner.encoder.video_text_encoder import TYPE_OUTPUT
from aligner.metrics import MedianRank, Rank
from aligner.video_text_module import TYPE_INPUT, VideoTextLightningModule
from util.tensor_utils import all_gather


class TextVideoRetrievalLightningModule(VideoTextLightningModule):  # noqa
    def __init__(self, *args, dataset_names: Optional[Iterable[str]] = None, compute_rank: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        metrics_dict = {"r1": Recall(), "r5": Recall(top_k=5), "r10": Recall(top_k=10), "mr": MedianRank()}

        if compute_rank:
            metrics_dict["rank"] = Rank()

        self.dataset_names = list(dataset_names) if dataset_names else None

        self.multiple_datasets = self.dataset_names is not None and len(self.dataset_names) > 1

        if self.multiple_datasets:
            assert all("_" not in name for name in self.dataset_names), \
                "Underscores in dataset names are problematic because of how we get their corresponding metrics."
            self.metrics: Mapping[str, Metric] = nn.ModuleDict((f"{name}_{dataset_name}", metric.clone())  # noqa
                                                               for dataset_name in self.dataset_names
                                                               for name, metric in metrics_dict.items())
        else:
            self.metrics: Mapping[str, Metric] = nn.ModuleDict(metrics_dict)

    @overrides(check_signature=False)
    def validation_step(self, batch: TYPE_INPUT, batch_idx: int = 0,
                        dataloader_idx: Optional[int] = None) -> Tuple[TYPE_OUTPUT, Optional[int]]:
        return self._step(batch, batch_idx), dataloader_idx

    def _validation_dataset_step_end(self, output: TYPE_OUTPUT, dataset_name: Optional[str] = None) -> TYPE_OUTPUT:
        encoded_video, encoded_text = all_gather(self, output)

        batch_size = len(encoded_video)

        logit_scale = self.logit_scale.exp()
        scores = logit_scale * encoded_video @ encoded_text.T
        loss = self.loss(scores)

        # Note that we need to pass the batch size in the first step log
        # as it can't be easily inferred by PL in our case.
        key = "loss/val" + ("" if dataset_name is None else f"_{dataset_name}")
        self.log(key, loss, prog_bar=True, batch_size=batch_size, add_dataloader_idx=False)

        return encoded_video, encoded_text

    @overrides(check_signature=False)
    def validation_step_end(self, output: Tuple[TYPE_OUTPUT, int]) -> TYPE_OUTPUT:
        step_output, data_loader_idx = output
        assert self.multiple_datasets == (data_loader_idx is not None)
        dataset_name = self.dataset_names[data_loader_idx] if self.multiple_datasets else None
        return self._validation_dataset_step_end(step_output, dataset_name=dataset_name)

    def _validate_dataset(self, outputs: Sequence[TYPE_OUTPUT], dataset_name: Optional[str] = None) -> None:
        assert self.multiple_datasets == (dataset_name is not None)

        encoded_videos, encoded_texts = (torch.cat(x) for x in zip(*outputs))

        batch_size = len(encoded_videos)

        scores = encoded_texts @ encoded_videos.T

        target = torch.arange(scores.shape[-1], device=scores.device)

        for name, metric in self.metrics.items():
            if not dataset_name or name.endswith(f"_{dataset_name}"):
                metric(scores, target)
                # Note that we need to pass the batch size in the first step log
                # as it can't be easily inferred by PL in our case.
                self.log(name, metric, prog_bar=True, batch_size=batch_size, add_dataloader_idx=False)

    @overrides(check_signature=False)
    def validation_epoch_end(self, outputs: Union[Sequence[TYPE_OUTPUT], Sequence[Sequence[TYPE_OUTPUT]]]) -> None:
        if self.multiple_datasets:
            for i, (name, dataset_output) in enumerate(zip(self.dataset_names, outputs)):
                # Necessary to set the current data loader ID so PL knows to which one the logged metrics belong
                # (because it returns the metrics by data loader).
                self._current_dataloader_idx = i
                self._validate_dataset(dataset_output, dataset_name=name)  # noqa
            self._current_dataloader_idx = None
        else:
            self._validate_dataset(outputs)

        if "rank" in self.metrics:
            self.print(self.metrics["rank"].compute().tolist())

    @overrides
    def load_state_dict(self, state_dict: "OrderedDict[str, torch.Tensor]", strict: bool = True):
        # If it's exactly this class, then ignore any teacher-related thing.
        # We do it here, so we can control it more, and avoid bugs with a general solution.
        if type(self) is TextVideoRetrievalLightningModule:
            incompatible_keys = super().load_state_dict(state_dict, strict=False)

            unexpected_keys = set(incompatible_keys.unexpected_keys)
            for key in incompatible_keys.unexpected_keys:
                if key.startswith("teacher"):
                    unexpected_keys.remove(key)

            # We then do as in super:

            if strict:
                error_msgs = []

                if unexpected_keys:
                    unexpected_key_str = ", ".join(f'"{k}"' for k in unexpected_keys)
                    error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected_key_str}. ")
                if incompatible_keys.missing_keys:
                    missing_keys_str = ', '.join(f'"{k}"' for k in incompatible_keys.missing_keys)
                    error_msgs.append(f"Missing key(s) in state_dict: {missing_keys_str}. ")

                if error_msgs:
                    error_msgs_str = "\n\t".join(error_msgs)
                    raise RuntimeError(f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t"
                                       f"{error_msgs_str}")

            return incompatible_keys
        else:
            return super().load_state_dict(state_dict, strict)
