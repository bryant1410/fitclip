import itertools
from typing import Iterable, Mapping, MutableMapping, Optional, Tuple, Union

import torch.distributed.nn
from overrides import overrides
from torch import nn

from aligner.encoder import video_text_encoder
from aligner.encoder.video_text_encoder import TYPE_TOKENIZER, VideoTextEncoder
from aligner.loss import TeacherStudentNCELoss
from aligner.text_video_retrieval import TextVideoRetrievalLightningModule
from aligner.video_text_module import TYPE_INPUT, TYPE_SPLIT, log_lr
from util.tensor_utils import all_gather, pad, split_in_collection

TYPE_OUTPUT = Tuple[video_text_encoder.TYPE_OUTPUT, video_text_encoder.TYPE_OUTPUT]

TYPE_MULTI_OUTPUT = Mapping[str, TYPE_OUTPUT]


def _replace_in_tokenized_text(tokenized_text: MutableMapping[str, torch.Tensor],
                               new_tokenized_text: Mapping[str, torch.Tensor], start_idx: int, end_idx: int,
                               tokenizer: TYPE_TOKENIZER) -> None:
    """Replaces the content in the tensor `tokenized_text` from the index `start_idx` to `end_idx` (exclusive) for
    `new_tokenized_text`.

    When it needs to know details about the tokenization, it uses `tokenizer`.
    """
    for k in tokenized_text:
        padding_value = 0 if "mask" in k else getattr(tokenizer, "pad_token_id", 0)

        # We suppose right padding.
        if tokenized_text[k].shape[1] > new_tokenized_text[k].shape[1]:
            padded = pad(new_tokenized_text[k], min_size=tokenized_text[k].shape[1], value=padding_value)
            tokenized_text[k] = torch.cat((tokenized_text[k][:start_idx], padded, tokenized_text[k][end_idx:]))
        elif tokenized_text[k].shape[1] < new_tokenized_text[k].shape[1]:
            padded = pad(tokenized_text[k], min_size=new_tokenized_text[k].shape[1], value=padding_value)
            tokenized_text[k] = torch.cat((padded[:start_idx], new_tokenized_text[k], padded[end_idx:]))
        else:
            tokenized_text[k] = torch.cat((tokenized_text[k][:start_idx], new_tokenized_text[k],
                                           tokenized_text[k][end_idx:]))


class TeacherStudentLightningModule(TextVideoRetrievalLightningModule):  # noqa
    """
    Distillation training module.

    If specified, `prompts` is used with the unlabeled dataset videos instead of the labels it provides (if any).
    """

    def __init__(self, encoder: VideoTextEncoder, teacher: VideoTextEncoder, labeled_dataset_name: str = "labeled",
                 labeled_dataset_loss_share: Optional[float] = None,
                 dataset_names: Iterable[str] = ("labeled", "unlabeled"), prompts: Optional[Iterable[str]] = None,
                 **kwargs) -> None:
        super().__init__(encoder=encoder, dataset_names=dataset_names, **kwargs)
        self.teacher = teacher

        assert self.dataset_names, "This module uses dataset names."
        assert len(self.dataset_names) == 2, "The current implementation needs exactly 2 datasets."
        # FIXME: it doesn't work with different datasets for training and evaluation, because it needs certain names
        #  for training; and this logic assumes the same dataset names for both.

        if labeled_dataset_loss_share is None:
            self.dataset_loss_share = {name: 1 / len(self.dataset_names) for name in self.dataset_names}
        else:
            self.dataset_loss_share = {labeled_dataset_name: labeled_dataset_loss_share}
            self.dataset_loss_share.update((name, (1 - labeled_dataset_loss_share) / (len(self.dataset_names) - 1))
                                           for name in self.dataset_names
                                           if name != labeled_dataset_name)

        self.teacher_student_logit_scale = nn.Parameter(self.logit_scale.clone(),
                                                        requires_grad=self.logit_scale.requires_grad)
        # noinspection SpellCheckingInspection
        self.teacher_student_loss = TeacherStudentNCELoss(reduction="batchmean")

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.labeled_dataset_name = labeled_dataset_name
        self.unlabeled_dataset_name = next(k for k in self.dataset_names if k != labeled_dataset_name)

        if prompts is None:
            self.tokenized_prompts = None
            self.teacher_tokenized_prompts = None
        else:
            prompts = list(prompts)
            # We use parameters so the device and dtype are moved correctly along with this module.
            self.tokenized_prompts = nn.ParameterDict((k, nn.Parameter(v, requires_grad=False))  # noqa
                                                      for k, v in encoder.get_tokenizer()(prompts).items())
            self.teacher_tokenized_prompts = nn.ParameterDict((k, nn.Parameter(v, requires_grad=False))  # noqa
                                                              for k, v in teacher.get_tokenizer()(prompts).items())

    @overrides(check_signature=False)
    def _step(self, batch: TYPE_INPUT, _batch_idx: int = 0) -> TYPE_OUTPUT:
        # Note we pass the labeled dataset portion to the teacher, but then we don't use it.
        return self({"video": batch["video_student"], "text": batch["text_student"]}), \
               self.teacher(video=batch["video_teacher"], text=batch["text_teacher"])

    @overrides(check_signature=False)
    def training_step(self, batch: TYPE_INPUT, _batch_idx: int = 0) -> TYPE_MULTI_OUTPUT:
        keys, lengths = zip(*((key, sum(1 for _ in group))
                              for key, group in itertools.groupby(dataset for dataset in batch.pop("dataset"))))
        assert len(keys) == len(self.dataset_names), "All datasets should be present in each batch."

        if self.tokenized_prompts is None:
            unlabeled_dataset_idx = None
        else:
            unlabeled_dataset_idx = keys.index(self.unlabeled_dataset_name)
            start_idx_in_batch = sum(lengths[i] for i in range(unlabeled_dataset_idx))
            end_idx_in_batch = start_idx_in_batch + lengths[unlabeled_dataset_idx]
            _replace_in_tokenized_text(tokenized_text=batch["text_student"],
                                       new_tokenized_text=self.tokenized_prompts, start_idx=start_idx_in_batch,
                                       end_idx=end_idx_in_batch, tokenizer=self.encoder.get_tokenizer())
            _replace_in_tokenized_text(tokenized_text=batch["text_teacher"],
                                       new_tokenized_text=self.teacher_tokenized_prompts,
                                       start_idx=start_idx_in_batch, end_idx=end_idx_in_batch,
                                       tokenizer=self.teacher.get_tokenizer())

        output = self._step(batch, _batch_idx)

        # Need to log the step because PL doesn't log it in Neptune.
        # See https://github.com/PyTorchLightning/pytorch-lightning/pull/5510
        first_video_value = next(v for k, v in batch.items() if k.startswith("video"))
        self.log(f"step", self.global_step, batch_size=len(first_video_value))

        if self.tokenized_prompts is None:
            split_output = split_in_collection(output, lengths)
        else:
            text_split_sections = list(lengths)
            text_split_sections[unlabeled_dataset_idx] = len(next(iter(self.tokenized_prompts.values())))

            student_video_sections = split_in_collection(output[0][0], lengths)
            student_text_sections = split_in_collection(output[0][1], text_split_sections)
            teacher_video_sections = split_in_collection(output[1][0], lengths)
            teacher_text_sections = split_in_collection(output[1][1], text_split_sections)

            split_output = (((student_video_sections[i], student_text_sections[i]),
                             (teacher_video_sections[i], teacher_text_sections[i]))
                            for i in range(len(student_video_sections)))

        return dict(zip(keys, split_output))

    def _dataset_step_end(self, output: TYPE_OUTPUT, split: TYPE_SPLIT,
                          dataset_name: Optional[str] = None) -> Union[torch.Tensor, video_text_encoder.TYPE_OUTPUT]:
        gathered_output = all_gather(self, output, sync_grads=split == "train")

        (encoded_video, encoded_text), (teacher_encoded_video, teacher_encoded_text) = gathered_output

        batch_size = len(encoded_video)

        logit_scale = self.logit_scale.exp()

        scores = logit_scale * encoded_video @ encoded_text.T

        if dataset_name == self.labeled_dataset_name:
            loss = self.loss(scores)
        else:
            teacher_student_logit_scale = self.teacher_student_logit_scale.exp()
            teacher_scores = teacher_student_logit_scale * teacher_encoded_video @ teacher_encoded_text.T
            loss = self.teacher_student_loss(scores, teacher_scores) * teacher_student_logit_scale ** 2

            if split == "train":
                # Note that we need to pass the batch size in the first step log
                # as it can't be easily inferred by PL in our case.
                self.log("batch_size", float(batch_size), batch_size=batch_size)
                self.log("temperature/labeled", 1 / logit_scale)
                self.log("temperature/unlabeled", 1 / teacher_student_logit_scale)

        prefix = f"loss/{split}_{dataset_name}" if dataset_name else f"loss/{split}"
        # Note that we need to pass the batch size in the first step log
        # as it can't be easily inferred by PL in our case.
        self.log(prefix, loss, prog_bar=split != "train", batch_size=batch_size, add_dataloader_idx=False)

        return loss if split == "train" else (encoded_video, encoded_text)

    @overrides(check_signature=False)
    def training_step_end(self, output: TYPE_MULTI_OUTPUT) -> torch.Tensor:
        loss = sum(self._dataset_step_end(batch, split="train", dataset_name=name) * self.dataset_loss_share[name]
                   for name, batch in output.items())
        self.log("loss/train", loss)  # Note train loss it's already shown in the progress bar by PL by default.

        log_lr(self)

        return loss

    @overrides(check_signature=False)
    def _validation_dataset_step_end(self, output: TYPE_OUTPUT,
                                     dataset_name: Optional[str] = None) -> video_text_encoder.TYPE_OUTPUT:
        return self._dataset_step_end(output, split="val", dataset_name=dataset_name)

    @overrides(check_signature=False)
    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        if self.teacher_student_logit_scale >= self.max_logit_scale:
            self.teacher_student_logit_scale.copy_(self.max_logit_scale)
