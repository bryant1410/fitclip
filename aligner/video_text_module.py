from typing import Any, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import math
import pytorch_lightning as pl
import torch.distributed.nn
from overrides import overrides
from torch import nn
from torch.nn.modules.loss import _Loss

from aligner.encoder.video_text_encoder import TYPE_OUTPUT, VideoTextEncoder
from aligner.loss import NCELoss
from util.tensor_utils import all_gather

TYPE_INPUT = MutableMapping[str, Any]
TYPE_SPLIT = Literal["train", "val"]


def log_lr(pl_module: pl.LightningModule, **kwargs) -> None:
    for i, optimizer in enumerate(pl_module.trainer.optimizers):
        for j, param_group in enumerate(optimizer.param_groups):
            if (lr := param_group.get("lr")) is not None:  # noqa
                pl_module.log(f"lr_{i}_group_{j}", lr, **kwargs)


class VideoTextLightningModule(pl.LightningModule):  # noqa
    def __init__(self, encoder: VideoTextEncoder, init_temperature: float = 0.05, min_temperature: float = 0.001,
                 fit_temperature: bool = True, loss: Optional[_Loss] = None) -> None:
        super().__init__()
        self.encoder = encoder

        # Use the temperature as in CLIP: save it in log-space and fit it along with the model.
        self.logit_scale = nn.Parameter(torch.tensor([- math.log(init_temperature)]), requires_grad=fit_temperature)
        # The following constant is set also as a parameter, so it's moved to the correct device automatically.
        self.max_logit_scale = nn.Parameter(torch.tensor([- math.log(min_temperature)]), requires_grad=False)
        self.loss = loss or NCELoss()

    @overrides(check_signature=False)
    def forward(self, batch: TYPE_INPUT,
                _batch_idx: int = 0) -> Union[TYPE_OUTPUT, Tuple[torch.Tensor, torch.Tensor, Sequence[str]]]:
        batch.pop("video_id", None)
        return self.encoder(**batch)

    def _step(self, batch: TYPE_INPUT, batch_idx: int = 0) -> TYPE_OUTPUT:
        return self(batch, batch_idx)

    @overrides(check_signature=False)
    def training_step(self, batch: TYPE_INPUT, _batch_idx: int = 0) -> TYPE_OUTPUT:
        output = self._step(batch, _batch_idx)
        # Need to log the step because PL doesn't log it in Neptune.
        # See https://github.com/PyTorchLightning/pytorch-lightning/pull/5510
        first_video_value = next(v for k, v in batch.items() if k.startswith("video"))
        self.log("step", float(self.global_step), batch_size=len(first_video_value))
        return output

    def _step_end(self, output: TYPE_OUTPUT, split: TYPE_SPLIT,
                  log_kwargs: Optional[Mapping[str, Any]] = None) -> Union[torch.Tensor, TYPE_OUTPUT]:
        log_kwargs = log_kwargs or {}
        encoded_video, encoded_text = all_gather(self, output, sync_grads=split == "train")

        batch_size = len(encoded_video)

        logit_scale = self.logit_scale.exp()
        scores = logit_scale * encoded_video @ encoded_text.T
        loss = self.loss(scores)

        # Note train loss it's already shown in the progress bar by PL by default.
        #
        # Note that we need to pass the batch size in the first step log
        # as it can't be easily inferred by PL in our case.
        self.log(f"loss/{split}", loss, prog_bar=split != "train", batch_size=batch_size, **log_kwargs)

        if split == "train":
            self.log("batch_size", float(batch_size), batch_size=batch_size)
            self.log("temperature", 1 / logit_scale, batch_size=batch_size)

        return loss if split == "train" else (encoded_video, encoded_text)

    @overrides(check_signature=False)
    def training_step_end(self, output: TYPE_OUTPUT) -> torch.Tensor:
        loss = self._step_end(output, split="train")
        log_lr(self)
        return loss

    @overrides(check_signature=False)
    def predict_step(self, batch: TYPE_INPUT, batch_idx: int = 0) -> Mapping[str, torch.Tensor]:
        encoded_video, encoded_text = self._step(batch, batch_idx)
        return {
            "encoded_videos": encoded_video,
            "encoded_texts": encoded_text,
            "video_ids": batch["video_id"]
        }

    @overrides(check_signature=False)
    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        if self.logit_scale >= self.max_logit_scale:
            self.logit_scale.copy_(self.max_logit_scale)
