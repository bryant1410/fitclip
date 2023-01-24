from typing import Literal

import torch
from overrides import overrides
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

TYPE_REDUCTION = Literal["none", "mean", "sum"]
# noinspection SpellCheckingInspection
TYPE_REDUCTION_KL_DIV = Literal["none", "batchmean", "mean", "sum"]


def _rows_to_columns_nce_loss(scores: torch.Tensor, reduction: TYPE_REDUCTION = "mean") -> torch.Tensor:
    loss = - F.log_softmax(scores, dim=-1).diag()

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def nce_loss(scores: torch.Tensor, reduction: TYPE_REDUCTION = "mean") -> torch.Tensor:
    return (_rows_to_columns_nce_loss(scores, reduction=reduction)
            + _rows_to_columns_nce_loss(scores.T, reduction=reduction))


def _rows_to_columns_teacher_student_nce_loss(scores: torch.Tensor, teacher_scores: torch.Tensor,
                                              reduction: TYPE_REDUCTION_KL_DIV = "mean") -> torch.Tensor:
    logits = F.log_softmax(scores, dim=-1)
    teacher_probs = F.softmax(teacher_scores, dim=-1)
    return F.kl_div(logits, teacher_probs, reduction=reduction)


def teacher_student_nce_loss(scores: torch.Tensor, teacher_scores: torch.Tensor,
                             reduction: TYPE_REDUCTION_KL_DIV = "mean") -> torch.Tensor:
    return (_rows_to_columns_teacher_student_nce_loss(scores, teacher_scores, reduction=reduction)
            + _rows_to_columns_teacher_student_nce_loss(scores.T, teacher_scores.T, reduction=reduction))


class NCELoss(_Loss):
    @overrides(check_signature=False)
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return nce_loss(scores, reduction=self.reduction)  # noqa


class TeacherStudentNCELoss(_Loss):
    @overrides(check_signature=False)
    def forward(self, scores: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
        return teacher_student_nce_loss(scores, teacher_scores, reduction=self.reduction)  # noqa


class SimilarityLoss(_Loss):
    @overrides(check_signature=False)
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        # Note we actually don't need all the scores.
        loss = - torch.log(torch.sigmoid(scores.diag()))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
