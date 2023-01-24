import torch
from overrides import overrides
from torchmetrics import Metric


class Rank(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("ranks", default=[], dist_reduce_fx="cat")

    @overrides(check_signature=False)
    def update(self, predictions: torch.Tensor, target: torch.Tensor) -> None:
        sorted_predicted_positions = predictions.argsort(dim=1, descending=True)
        ranks = torch.where(sorted_predicted_positions == target.unsqueeze(-1))[1]  # noqa
        self.ranks.append(ranks)

    @overrides
    def compute(self) -> torch.Tensor:
        # It could be already reduced depending on when we call it (e.g., at the epoch end).
        return self.ranks if isinstance(self.ranks, torch.Tensor) else torch.cat(self.ranks)


class MeanRank(Rank):
    @overrides
    def compute(self) -> torch.Tensor:
        return super().compute().mean() + 1


class MedianRank(Rank):
    @overrides
    def compute(self) -> torch.Tensor:
        return super().compute().median() + 1
