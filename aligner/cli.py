#!/usr/bin/env python
import copy
import logging
import warnings
from types import MethodType
from typing import Any, Mapping, Optional, Tuple, Type

import hydra
import pytorch_lightning as pl
from cached_path import cached_path
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from torch.optim import Optimizer

from aligner.data.data_module_group import DataModuleStructuredGroup, EvalDataModuleGroup, MixedBatchDataModule, \
    TrainAndEvalDataModules
from aligner.data.video_data_module import ENCODER_OR_ENCODER_MAP, VideoClassificationDataModule
from aligner.encoder.video_text_encoder import VideoTextEncoder
from aligner.video_text_classification import VideoTextClassificationLightningModule
from aligner.video_text_module import VideoTextLightningModule

LOGGER = logging.getLogger(__name__)

# This is because PL can't automatically infer the batch size, that's needed for logging. But we set it manually
# within the modules.
warnings.filterwarnings("ignore", message=r"^Trying to infer the `batch_size` from an ambiguous collection\. .+")


# From https://stackoverflow.com/a/2020083/1165181
def fullname(klass: Type[Any]) -> str:
    return f"{klass.__module__}.{klass.__qualname__}"


def set_logging_level(level: int) -> None:
    logging.basicConfig(level=level)
    # `basicConfig` will only work for new loggers, so we also need to set up the existing ones:
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):  # Otherwise, it could be a `logging.PlaceHolder`.
            logger.setLevel(level)
    logging.getLogger().setLevel(level)  # The root logger is not present in the previous iterable.


def init_cli(cfg: DictConfig) -> None:
    if cfg.get("silent"):
        set_logging_level(logging.WARNING)
    else:
        set_logging_level(logging.INFO)

    if "seed" in cfg:
        seed_everything(cfg.seed, workers=True)


def instantiate_data_module(cfg: DictConfig, encoder: ENCODER_OR_ENCODER_MAP) -> pl.LightningDataModule:
    kwargs = {}

    if cfg._target_ in {fullname(klass) for klass in [DataModuleStructuredGroup, EvalDataModuleGroup,
                                                      MixedBatchDataModule]}:
        if isinstance(cfg.data_modules, Mapping):
            kwargs["data_modules"] = {k: instantiate_data_module(v, encoder=encoder)  # noqa
                                      for k, v in cfg.data_modules.items()}
        else:
            kwargs["data_modules"] = {instantiate_data_module(cfg_dm, encoder=encoder)
                                      for cfg_dm in cfg.data_modules}

        # Convert because otherwise the passed `data_modules` is a `DictConfig` instead of a `dict` and
        # `train_dataloader` can't respect the same collection type as `DictConfig` can't have normal classes.
        kwargs["_convert_"] = "all"
    elif cfg._target_ == fullname(TrainAndEvalDataModules):
        kwargs["train_data_module"] = instantiate_data_module(cfg.train_data_module, encoder=encoder)

        kwargs["eval_data_module"] = instantiate_data_module(cfg.eval_data_module, encoder=encoder)
    else:
        kwargs["encoder"] = encoder

        # Necessary as well when the encoder is a dict.
        kwargs["_convert_"] = "all"

    return hydra.utils.instantiate(cfg, **kwargs)


def create_model_data_module_trainer_and_ckpt_path(
        cfg: DictConfig, model_kwargs: Optional[Mapping[str, Any]] = None) -> Tuple[VideoTextLightningModule,
                                                                                    pl.LightningDataModule, pl.Trainer,
                                                                                    str]:
    model_kwargs = model_kwargs or {}

    LOGGER.info(f"Instantiating encoder <{getattr(cfg.encoder, '_target_', type(cfg.encoder).__name__)}>…")

    encoder: ENCODER_OR_ENCODER_MAP = hydra.utils.instantiate(cfg.encoder)

    if isinstance(encoder, Mapping) and cfg.get("use_student_encoder_for_data_preprocessing"):
        encoder_for_data_preprocessing = encoder["student"]
    else:
        encoder_for_data_preprocessing = encoder

    LOGGER.info("Encoder instantiated.")

    LOGGER.info(f"Instantiating data module <{cfg.data._target_}>…")
    data_module = instantiate_data_module(cfg.data, encoder=encoder_for_data_preprocessing)
    LOGGER.info("Data module instantiated.")

    LOGGER.info(f"Instantiating model <{cfg.model._target_}>…")

    if isinstance(encoder, Mapping):
        model_kwargs.setdefault("encoder", encoder["student"])
        model_kwargs.setdefault("teacher", encoder["teacher"])
    else:
        model_kwargs.setdefault("encoder", encoder)

    if isinstance(data_module, VideoClassificationDataModule):
        assert isinstance(encoder_for_data_preprocessing, VideoTextEncoder), \
            "Encoder can't be a mapping and has to support text when doing classification."
        cfg.model._target_ = fullname(VideoTextClassificationLightningModule)
        model_kwargs.setdefault("labels", data_module.categories)
        model_kwargs.setdefault("templates", data_module.templates)

    if prompts_path := cfg.get("prompts"):  # noqa
        with open(cached_path(prompts_path)) as file:
            model_kwargs.setdefault("prompts", [stripped_line
                                                for line in file
                                                if (stripped_line := line.strip())])  # noqa

    model: VideoTextLightningModule = hydra.utils.instantiate(cfg.model, **model_kwargs)
    LOGGER.info("Model instantiated.")

    if "optimizer" in cfg:
        LOGGER.info(f"Instantiating Optimizer <{cfg.optimizer._target_}>…")

        def configure_optimizers(self: pl.LightningModule) -> Optimizer:
            if (lr_ := self.hparams.get("lr")) is not None:  # To be used by auto LR find.
                cfg.optimizer["lr"] = lr_
            return hydra.utils.instantiate(cfg.optimizer, self.parameters())

        model.configure_optimizers = MethodType(configure_optimizers, model)
        LOGGER.info("Optimizer instantiated.")

    LOGGER.info(f"Instantiating trainer <{cfg.trainer._target_}>…")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    LOGGER.info("Trainer instantiated.")

    # We do what `model.save_hyperparameters(cfg)` would do but without needing a current frame to get the args from.
    # It turns out that, even if you provide args, it still checks the current frame for args, and set those
    # conditioned by the provided args.
    model._log_hyperparams = trainer.logger
    model._set_hparams(cfg)  # noqa
    model._hparams_initial = copy.deepcopy(model._hparams)

    ckpt_path = cached_path(cfg.checkpoint_path) if cfg.get("path") else None

    return model, data_module, trainer, ckpt_path
