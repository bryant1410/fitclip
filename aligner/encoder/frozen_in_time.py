# Originally from https://github.com/m-bain/frozen-in-time/blob/ba54e43/model/model.py
import logging
import sys
from typing import Any, Dict, Literal, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from cached_path import TYPE_PATH, cached_path
from transformers import AutoModel

from aligner.encoder import frozen_in_time_stub
from aligner.encoder.video_transformer import SpaceTimeTransformer

LOGGER = logging.getLogger(__name__)

STATE_DICT_MODULE_KEY = "module."


def state_dict_data_parallel_fix(load_state_dict: MutableMapping[str, Any],
                                 curr_state_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    first_load_key = next(iter(load_state_dict.keys()))
    first_curr_key = next(iter(curr_state_dict.keys()))

    if not first_curr_key.startswith(STATE_DICT_MODULE_KEY) and first_load_key.startswith(STATE_DICT_MODULE_KEY):
        return {k[len(STATE_DICT_MODULE_KEY):]: v for k, v in load_state_dict.items()}
    elif first_curr_key.startswith(STATE_DICT_MODULE_KEY) and not first_load_key.startswith(STATE_DICT_MODULE_KEY):
        return {STATE_DICT_MODULE_KEY + k: v for k, v in load_state_dict.items()}
    else:
        return load_state_dict


class BaseModel(nn.Module):
    """Base class for all models"""

    def __str__(self) -> str:
        return f"{super().__str__()}\n" \
               f"Trainable parameters: {sum(np.prod(p.size()) for p in self.parameters() if p.requires_grad)}"


class FrozenInTime(BaseModel):
    def __init__(self, video_params: Dict[str, Any], text_params: Dict[str, Any], projection_dim: int = 256,
                 load_checkpoint: Optional[TYPE_PATH] = None, projection: Literal["", "minimal"] = "minimal",
                 load_temporal_fix: Literal["zeros", "interp", "bilinear"] = "zeros") -> None:
        super().__init__()

        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        if not text_params["pretrained"]:
            raise ValueError("HuggingFace text models require `pretrained` init.")

        transformers_modeling_utils_logger = logging.getLogger("transformers.modeling_utils")
        transformers_modeling_utils_logger.disabled = True
        self.text_model = AutoModel.from_pretrained(text_params["model"])
        transformers_modeling_utils_logger.disabled = False

        pretrained = video_params["pretrained"]
        if video_params["model"] == "SpaceTimeTransformer":
            num_frames = video_params.get("num_frames", 4)
            time_init = video_params.get("time_init", "zeros")
            attention_style = video_params.get("attention_style", "frozen-in-time")
            arch_config = video_params.get("arch_config", "base_patch16_224")
            if arch_config == "base_patch16_224":
                vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                model = SpaceTimeTransformer(num_frames=num_frames, time_init=time_init,
                                             attention_style=attention_style)
            else:
                raise ValueError(f"Unrecognized arch_config: {arch_config}")

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
            if not load_checkpoint:
                vit_checkpoint = vit_model.state_dict()
                model.load_state_dict(vit_checkpoint, strict=False)
            self.video_model = model
        else:
            raise ValueError(f"{video_params['model']} not supported")

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        # Project to a common embedding
        if projection == "minimal":
            txt_proj = nn.Sequential(nn.ReLU(), nn.Linear(self.text_model.config.hidden_size, projection_dim))
            vid_proj = nn.Sequential(nn.Linear(ftr_dim, projection_dim))
        elif projection == "":
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise ValueError(f"Unrecognized projection: {projection}")

        self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        if load_checkpoint:
            load_checkpoint = cached_path(load_checkpoint)

            # To make pickle work with a missing module and class. See https://stackoverflow.com/a/2121918/1165181
            sys.modules["parse_config"] = frozen_in_time_stub

            LOGGER.info("Loading frozen-in-time checkpoint…")
            # `map_location="cpu"` to avoid bloating GPU=0 with each process' copy of it.
            checkpoint = torch.load(load_checkpoint, map_location="cpu")

            del sys.modules["parse_config"]

            state_dict = checkpoint["state_dict"]
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=True)  # noqa
            LOGGER.info("Checkpoint loaded.")

    def forward(self, data: Mapping[str, Any], return_embeds: bool = True) -> Union[torch.Tensor,
                                                                                    Tuple[torch.Tensor, torch.Tensor]]:
        text_data = data["text"]
        video_data = data["video"]

        text_embeddings = self.compute_text(text_data)
        video_embeddings = self.compute_video(video_data)

        if return_embeds:
            return text_embeddings, video_embeddings

        return sim_matrix(text_embeddings, video_embeddings)

    def compute_text(self, text_data: Mapping[str, Any]) -> torch.Tensor:
        if self.text_params["model"].startswith("bert"):
            text_embeddings = self.text_model(text_data["input_ids"], attention_mask=text_data["attention_mask"])[
                "pooler_output"]
        elif self.text_params["model"].startswith("distilbert"):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unrecognized text model: {self.text_params['model']}")
        return self.txt_proj(text_embeddings)

    def compute_video(self, video_data: Mapping[str, Any]) -> torch.Tensor:
        video_embeddings = self.video_model(video_data)
        return self.vid_proj(video_embeddings)

    def _inflate_positional_embeds(self, new_state_dict: MutableMapping[str, Any]) -> Mapping[str, Any]:
        # allow loading of timesformer with fewer num_frames
        curr_keys = set(self.state_dict().keys())
        if "video_model.temporal_embed" in new_state_dict and "video_model.temporal_embed" in curr_keys:
            load_temporal_embed = new_state_dict["video_model.temporal_embed"]
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params["num_frames"]
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    LOGGER.warning(f"The loaded {self.video_params['model']} model has MORE frames than the current "
                                   f"one. Loading weights, filling in the extras via {self.load_temporal_fix}")
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    LOGGER.warning(f"The loaded {self.video_params['model']} model has FEWER frames than the current "
                                   f"one. Loading weights, filling in the extras via {self.load_temporal_fix}")
                    if self.load_temporal_fix == "zeros":
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ["interp", "bilinear"]:
                        # interpolate
                        # unsqueeze so pytorch thinks it's an image
                        mode = "nearest"
                        if self.load_temporal_fix == "bilinear":
                            mode = "bilinear"
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                    else:
                        raise ValueError(f"Unrecognized load_temporal_fix: {self.load_temporal_fix}")
                new_state_dict["video_model.temporal_embed"] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if "video_model.pos_embed" in new_state_dict and "video_model.pos_embed" in curr_keys:
            load_pos_embed = new_state_dict["video_model.pos_embed"]
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()["video_model.pos_embed"]
            if load_num_patches != curr_pos_embed.shape[1]:
                raise ValueError(
                    "Loading models with different spatial resolution / patch number not yet implemented, sorry.")

        return new_state_dict


def sim_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch:
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))  # noqa
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))  # noqa
    return torch.mm(a_norm, b_norm.transpose(0, 1))
