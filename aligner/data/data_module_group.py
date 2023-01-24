import bisect
from abc import ABC
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import pytorch_lightning as pl
from overrides import overrides
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.datasets.samplers import DistributedSampler as DistributedSampler2

from aligner.data.multi_source_sampler import RoundRobinMultiSourceSampler

TYPE_DM_ITERABLE_OR_MAP = Union[Iterable[pl.LightningDataModule], Mapping[str, pl.LightningDataModule]]


def _data_modules_iterable(data_modules: TYPE_DM_ITERABLE_OR_MAP) -> Iterable[pl.LightningDataModule]:
    return data_modules.values() if isinstance(data_modules, Mapping) else data_modules


def _data_loader_sequence(data_modules: TYPE_DM_ITERABLE_OR_MAP,
                          fn: Callable[[pl.LightningDataModule], EVAL_DATALOADERS]) -> Sequence[DataLoader]:
    dls = (fn(dm) for dm in _data_modules_iterable(data_modules))
    return [dl for dls_dm in dls for dl in ([dls_dm] if isinstance(dls_dm, DataLoader) else dls_dm)]


class _DataModuleGroup(pl.LightningDataModule, ABC):
    def __init__(self, data_modules: TYPE_DM_ITERABLE_OR_MAP) -> None:
        # Before calling super because it sets `trainer`, which recursively uses these.
        self.data_modules = data_modules
        self._trainer = None
        super().__init__()

    # Use it as a property, so we can set it to the data modules when set to self.
    @property
    def trainer(self) -> Trainer:
        return self._trainer

    @trainer.setter
    def trainer(self, value: Trainer) -> None:
        self._trainer = value
        # `self.trainer` is set during `super().__init__`, which in turn it's called from `super().__new__`,
        # which we can't control and happens before `self.data_modules` even exists.
        # So we need to handle the case where the attribute doesn't exist.
        for dm in _data_modules_iterable(getattr(self, "data_modules", [])):
            dm.trainer = value

    @overrides
    def prepare_data(self) -> None:
        for dm in _data_modules_iterable(self.data_modules):
            dm.prepare_data()

    @overrides
    def setup(self, stage: Optional[str] = None) -> None:
        for dm in _data_modules_iterable(self.data_modules):
            dm.setup(stage)


class EvalDataModuleGroup(_DataModuleGroup):  # noqa
    @overrides
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return _data_loader_sequence(self.data_modules, lambda dm: dm.val_dataloader())

    @overrides
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return _data_loader_sequence(self.data_modules, lambda dm: dm.test_dataloader())

    @overrides
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return _data_loader_sequence(self.data_modules, lambda dm: dm.predict_dataloader())


class DataModuleStructuredGroup(EvalDataModuleGroup):
    @overrides
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return apply_to_collection(self.data_modules, pl.LightningDataModule, lambda dm: dm.train_dataloader())


class ConcatDatasetWithDatasetKey(ConcatDataset):
    """A `ConcatDataset` that returns the corresponding dataset key for each item.

    It supposes the underlying datasets all return mapping items.
    """

    def __init__(self, datasets: Union[Iterable[Dataset], Mapping[str, Dataset]]) -> None:
        super().__init__(datasets.values() if isinstance(datasets, Mapping) else datasets)
        self.keys = list(datasets.keys()) if isinstance(datasets, Mapping) else range(len(self.datasets))

    @overrides(check_signature=False)
    def __getitem__(self, i: int) -> Mapping[Any, Any]:
        item = super().__getitem__(i)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, i)
        return {**item, "dataset": self.keys[dataset_idx]}


def _add_distributed_sampler(data_loaders: EVAL_DATALOADERS, mode: RunningStage) -> EVAL_DATALOADERS:
    assert all(apply_to_collection(data_loaders, DataLoader, lambda dl: isinstance(dl.sampler, SequentialSampler)))
    return apply_to_collection(
        data_loaders, DataLoader,
        lambda dl: Trainer._update_dataloader(dl, DistributedSampler2(dl.dataset), mode=mode))


class MixedBatchDataModule(EvalDataModuleGroup):
    """A data module that combines many data modules during training, with the same dataset composition for each batch,
    but separately for evaluation."""

    def __init__(self, *args, train_sequence_sizes: Union[int, Iterable[int], Mapping[str, int]] = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if isinstance(train_sequence_sizes, Mapping):
            assert isinstance(self.data_modules, Mapping)
            self.train_sequence_sizes = [train_sequence_sizes[k] for k in self.data_modules]
        else:
            self.train_sequence_sizes = train_sequence_sizes

        if isinstance(self.train_sequence_sizes, int):
            self.train_batch_size = len(self.data_modules) * self.train_sequence_sizes
        else:
            self.train_batch_size = sum(self.train_sequence_sizes)

    @overrides
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        data_loaders = apply_to_collection(self.data_modules, pl.LightningDataModule, lambda dm: dm.train_dataloader())
        datasets = apply_to_collection(data_loaders, DataLoader, lambda dl: dl.dataset)
        dataset = ConcatDatasetWithDatasetKey(datasets)

        sub_samplers = [RandomSampler(dataset) for dataset in dataset.datasets]  # noqa
        sampler = RoundRobinMultiSourceSampler(sub_samplers, sequence_sizes=self.train_sequence_sizes,
                                               mode="max_size_cycle")

        data_loader_iterable = data_loaders.values() if isinstance(data_loaders, Mapping) else data_loaders
        # We suppose each data module has the same args for the train data loader creation for the values obtained
        # here from the first data loader.
        first_data_loader = next(iter(data_loader_iterable))

        # We have to create the batch sampler manually for the distributed setting.
        # This is because we need to control how each batch is formed. If we don't do this, the distributed sampler
        # comes before the batch sampling, and the mix composition of the batches won't be the intended one.
        #
        # For simplicity, we apply it regardless of distributed/non-distributed setup.
        batch_sampler = BatchSampler(sampler, batch_size=self.train_batch_size, drop_last=True)

        if self.trainer._accelerator_connector.is_distributed:
            # We need to manually set the distributed sampler instead of doing it automatically with Pytorch Lightning
            # because we're using a custom sampler.
            #
            # This version of DistributedSampler accounts for having a sampler as input.
            #
            # BTW, there's a similar one (`DistributedSamplerWrapper`) in
            # https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
            batch_sampler = DistributedSampler2(batch_sampler)

        # We need to set the sampler as a `batch_sampler` so it activates the auto-collation in the data loader.
        data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=first_data_loader.num_workers,
                                 collate_fn=first_data_loader.collate_fn, pin_memory=first_data_loader.pin_memory,
                                 timeout=first_data_loader.timeout, worker_init_fn=first_data_loader.worker_init_fn,
                                 multiprocessing_context=first_data_loader.multiprocessing_context,
                                 prefetch_factor=first_data_loader.prefetch_factor,
                                 persistent_workers=first_data_loader.persistent_workers)

        if self.trainer._accelerator_connector.is_distributed:
            # PL only sets the epoch to the sampler, not to the batch sampler. This is because the distributed
            # sampler is typically the former not the latter.
            # Note that setting the epoch is necessary for shuffling, so every epoch has different batches.
            data_loader.sampler.set_epoch = lambda epoch: batch_sampler.set_epoch(epoch)

        return data_loader

    def _add_distributed_sampler_maybe(self, data_loaders: EVAL_DATALOADERS, mode: RunningStage) -> EVAL_DATALOADERS:
        if self.trainer._accelerator_connector.is_distributed:
            return _add_distributed_sampler(data_loaders, mode)
        else:
            return data_loaders

    @overrides
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._add_distributed_sampler_maybe(super().val_dataloader(), RunningStage.VALIDATING)

    @overrides
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._add_distributed_sampler_maybe(super().test_dataloader(), RunningStage.TESTING)

    @overrides
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self._add_distributed_sampler_maybe(super().predict_dataloader(), RunningStage.PREDICTING)


class TrainAndEvalDataModules(_DataModuleGroup):
    def __init__(self, train_data_module: pl.LightningDataModule, eval_data_module: pl.LightningDataModule) -> None:
        super().__init__([train_data_module, eval_data_module])

    @overrides
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.data_modules[0].train_dataloader()  # noqa

    @overrides
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.data_modules[1].val_dataloader()  # noqa

    @overrides
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.data_modules[1].test_dataloader()  # noqa

    @overrides
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.data_modules[1].predict_dataloader()  # noqa
