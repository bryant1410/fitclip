#!/usr/bin/env python
import logging
import os
from time import strftime
from typing import Mapping, Optional

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from aligner.cli import create_model_data_module_trainer_and_ckpt_path, init_cli
from aligner.logger_utils import get_logger_by_type

# Note it's better to have this as a module, so it's importable and DDP works fine in debug mode.
# Maybe this issue is caused by Hydra moving the CWD to somewhere else.

LOGGER = logging.getLogger(__name__)


# Set an env var, if empty, to the desired working directory in sweep mode. Then we read it from the config.
# This way we make sure all processes use the same folder.
# See https://github.com/PyTorchLightning/pytorch-lightning/issues/2727
os.environ.setdefault("SWEEP_DIR", f"multirun/{strftime('%Y-%m-%d')}/{strftime('%H-%M-%S')}")


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig) -> Optional[float]:
    init_cli(cfg)

    if cfg.get("trainer", {}).get("strategy") == "dp":
        LOGGER.warning("DP strategy not supported by the current metric logging scheme."
                       " See https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#logging-torchmetrics")

    model, data_module, trainer, ckpt_path = create_model_data_module_trainer_and_ckpt_path(cfg)

    output = None

    if cfg.command == "train":
        if cfg.get("validate_before_training"):
            LOGGER.info("Validation before training started.")
            with torch.inference_mode():
                metrics_list = trainer.validate(model, datamodule=data_module, ckpt_path=ckpt_path)
            LOGGER.info("Validation before training finished.")

            if (tb_logger := get_logger_by_type(trainer, TensorBoardLogger)) and not tb_logger._default_hp_metric:
                tb_logger.log_hyperparams(model.hparams_initial, metrics={k: v for metrics in metrics_list
                                                                          for k, v in metrics.items()})

        LOGGER.info("Training started.")
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

        if optimized_metric_name := cfg.get("optimized_metric_name"):
            output = trainer.callback_metrics.get(optimized_metric_name)
    elif cfg.command == "tune":
        assert ckpt_path is None, "Checkpoint path not supported when tuning."

        if trainer._accelerator_connector.is_distributed:
            LOGGER.warning("Tuning with the PL Trainer is known to have some issues in distributed settings."
                           " See e.g. https://github.com/PyTorchLightning/pytorch-lightning/issues/4280")

        LOGGER.info("Tuning started.")
        trainer.tune(model, datamodule=data_module)
    elif cfg.command in {"evaluate", "validate"}:
        with torch.inference_mode():
            trainer.validate(model, datamodule=data_module, ckpt_path=ckpt_path)
    elif cfg.command == "test":
        with torch.inference_mode():
            trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    elif cfg.command == "predict":
        if trainer._accelerator_connector.is_distributed:
            LOGGER.warning("Predicting with the PL Trainer is known to have some issues in distributed settings."
                           " See e.g. https://github.com/PyTorchLightning/pytorch-lightning/issues/10618")

        output_path = cfg.get("output_path", "predictions.pt")

        with torch.inference_mode():
            predictions = trainer.predict(model, datamodule=data_module, ckpt_path=ckpt_path)

        assert predictions
        first_prediction = predictions[0]

        assert isinstance(first_prediction, Mapping)
        keys = first_prediction

        predictions_map = {k: torch.cat([prediction[k] for prediction in predictions])
                           if isinstance(first_prediction[k], torch.Tensor)
                           else [p for prediction in predictions for p in prediction[k]]
                           for k in keys}

        torch.save(predictions_map, output_path)
    else:
        raise ValueError(f"Unrecognized command: {cfg.command}")

    if (neptune_logger := get_logger_by_type(trainer, NeptuneLogger)) and trainer.is_global_zero:
        # In a Hydra multirun (sweep) scenario, Neptune experiments from finished runs are marked as still running
        # unless we stop them manually. See https://github.com/PyTorchLightning/pytorch-lightning/issues/11368
        neptune_logger.run.stop()

    # Return the optimized metric value for hparam search.
    return output


if __name__ == "__main__":
    main()
