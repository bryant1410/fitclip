# Inspired from https://github.com/allenai/allennlp/blob/0d8c0fc/allennlp/training/optimizers.py
import logging
import re
from typing import Iterable, Optional, Union

import pytorch_lightning as pl
from overrides import overrides

LOGGER = logging.getLogger(__name__)


class ParamFreezer(pl.Callback):
    def __init__(self, regexes: Iterable[Union[str, re.Pattern]]) -> None:
        super().__init__()
        self.regexes = [re.compile(regex) for regex in regexes]

    @overrides
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        unused_regexes = {p.pattern for p in self.regexes}

        params_to_tune = []
        frozen_params = []

        for name, param in pl_module.named_parameters():
            for regex in self.regexes:
                if regex.search(name):
                    param.requires_grad = False

                    if regex.pattern in unused_regexes:
                        unused_regexes.remove(regex.pattern)

                    frozen_params.append(name)
                    break
            else:
                params_to_tune.append(name)

        LOGGER.debug(f"Params to tune: {params_to_tune}")
        LOGGER.debug(f"Frozen params: {frozen_params}")

        if unused_regexes:
            LOGGER.warning(f"The following param regexes used for freezing didn't match any param name: "
                           f"{unused_regexes}")
