import torch
import torch.utils.checkpoint
from torch import nn
from transformers import AutoConfig, BertModel, BertPreTrainedModel
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder


class VideoTokenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.input_dim if hasattr(config, "input_dim") else 512
        self.linear1 = nn.Linear(input_dim, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class MMBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.max_video_len = config.max_video_len
        if hasattr(config, "use_seg_emb") and config.use_seg_emb:
            """the original VLM paper uses seg_embeddings for temporal space.
            although not used it changed the randomness of initialization.
            we keep it for reproducibility.
            """
            self.seg_embeddings = nn.Embedding(256, config.hidden_size)

    def forward(  # noqa
            self,
            input_ids,
            input_video_embeds,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
    ):
        input_tensor = input_ids if input_ids is not None else inputs_embeds
        if input_video_embeds is not None:
            input_shape = (
                input_tensor.size(0),
                input_tensor.size(1) + input_video_embeds.size(1),
            )
        else:
            input_shape = (input_tensor.size(0), input_tensor.size(1))

        if position_ids is None:
            """
            Auto skip position embeddings for text only case.
            use cases:
            (1) action localization and segmentation:
                feed in len-1 dummy video token needs text part to
                skip input_video_embeds.size(1) for the right
                position_ids for video [SEP] and rest text tokens.
            (2) MMFusionShare for two forward passes:
                in `forward_text`: input_video_embeds is None.
                    need to skip video [SEP] token.
            # video_len + 1: [CLS] + video_embed
            # self.max_video_len + 1: [SEP] for video.
            # self.max_video_len + 2: [SEP] for video.
            # self.max_video_len + input_ids.size(1): rest for text.
            """
            if input_video_embeds is not None:
                video_len = input_video_embeds.size(1)
                starting_offset = self.max_video_len + 1  # video [SEP]
                ending_offset = self.max_video_len + input_ids.size(1)
            else:
                video_len = 0
                starting_offset = self.max_video_len + 2  # first text token.
                ending_offset = self.max_video_len + input_ids.size(1) + 1
            position_ids = torch.cat([
                self.position_ids[:, :video_len + 1],
                self.position_ids[:, starting_offset:ending_offset]
            ], dim=1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        """
        the format of input_ids is [CLS] [SEP] caption [SEP] padding.
        the goal is to build [CLS] video tokens [SEP] caption [SEP] .
        """
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if input_video_embeds is not None:
            inputs_mm_embeds = torch.cat([
                inputs_embeds[:, :1], input_video_embeds, inputs_embeds[:, 1:]
            ], dim=1)
        else:
            # text only for `MMFusionShare`.
            inputs_mm_embeds = inputs_embeds

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_mm_embeds + position_embeddings
        embeddings += token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiLayerAttentionMaskBertEncoder(BertEncoder):
    """extend BertEncoder with the capability of
    multiple layers of attention mask."""

    def forward(  # noqa
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_attention_mask = (
                attention_mask[:, i, :, :, :]
                if attention_mask.dim() == 5
                else attention_mask
            )

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    layer_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_attentions]
            if v is not None
        )


class MMBertModel(BertModel):
    """MMBertModel has MMBertEmbedding to support video tokens."""

    def __init__(self, config, add_pooling_layer=True):  # noqa
        super().__init__(config)
        # overwrite embedding
        self.embeddings = MMBertEmbeddings(config)
        self.encoder = MultiLayerAttentionMaskBertEncoder(config)
        self.init_weights()  # noqa

    def forward(
            self,
            input_ids=None,
            input_video_embeds=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            separate_forward_split=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions  # noqa
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states  # noqa
        )
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict  # noqa
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids "
                "and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            if input_video_embeds is not None:
                input_shape = (
                    input_ids.size(0),
                    input_ids.size(1) + input_video_embeds.size(1),
                )
            else:
                input_shape = (
                    input_ids.size(0),
                    input_ids.size(1),
                )
        elif inputs_embeds is not None:
            if input_video_embeds is not None:
                input_shape = (
                    inputs_embeds.size(0),
                    inputs_embeds.size(1) + input_video_embeds.size(1),
                )
            else:
                input_shape = (
                    input_ids.size(0),
                    input_ids.size(1),
                )
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = (inputs_embeds if input_ids is None else input_ids).device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case
        # we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = \
            self.get_extended_attention_mask(
                attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to
        # [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:  # noqa
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(  # noqa
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or
        # [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]

        head_mask = self.get_head_mask(  # noqa
            head_mask, self.config.num_hidden_layers)  # noqa

        embedding_output = self.embeddings(
            input_ids,
            input_video_embeds,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        if separate_forward_split is not None:
            split_embedding_output = \
                embedding_output[:, :separate_forward_split]
            split_extended_attention_mask = extended_attention_mask[
                                            :, :, :, :separate_forward_split, :separate_forward_split
                                            ]
            split_encoder_outputs = self.encoder(
                split_embedding_output,
                attention_mask=split_extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            assert (
                    len(split_encoder_outputs) <= 2
            ), "we do not support merge on attention for now."
            encoder_outputs = [[split_encoder_outputs[0]]]
            if len(split_encoder_outputs) == 2:
                encoder_outputs.append([])
                for _all_hidden_states in split_encoder_outputs[1]:
                    encoder_outputs[-1].append([_all_hidden_states])

            split_embedding_output = \
                embedding_output[:, separate_forward_split:]
            split_extended_attention_mask = extended_attention_mask[
                                            :, :, :, separate_forward_split:, separate_forward_split:
                                            ]

            split_encoder_outputs = self.encoder(
                split_embedding_output,
                attention_mask=split_extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            assert len(split_encoder_outputs) <= 2, "we do not support merge on attention for now."
            encoder_outputs[0].append(split_encoder_outputs[0])
            encoder_outputs[0] = torch.cat(encoder_outputs[0], dim=1)
            if len(split_encoder_outputs) == 2:
                for layer_idx, _all_hidden_states in enumerate(
                        split_encoder_outputs[1]
                ):
                    encoder_outputs[1][layer_idx].append(_all_hidden_states)
                    encoder_outputs[1][layer_idx] = torch.cat(
                        encoder_outputs[1][layer_idx], dim=1
                    )
            encoder_outputs = tuple(encoder_outputs)
        else:
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = encoder_outputs[0]
        pooled_output = None if self.pooler is None else self.pooler(sequence_output)  # noqa

        return (sequence_output, pooled_output) + encoder_outputs[1:]

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """This is borrowed from `modeling_utils.py` with the support of
        multi-layer attention masks.
        The second dim is expected to be number of layers.
        See `MMAttentionMaskProcessor`.
        Makes broadcastable attention and causal masks so that future
        and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to,
                zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.
        Returns:
            :obj:`torch.Tensor` The extended attention mask, with the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length] ourselves
        # in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 4:
            extended_attention_mask = attention_mask[:, :, None, :, :]
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # noqa; fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            return extended_attention_mask
        else:
            return super().get_extended_attention_mask(attention_mask, input_shape, device)  # noqa


class MMBertForEncoder(BertPreTrainedModel):
    """A BertModel for Contrastive Learning."""

    def __init__(self, config):
        super().__init__(config)
        self.videomlp = VideoTokenMLP(config)
        self.bert = MMBertModel(config)
        self.init_weights()  # noqa

    def forward(
            self,
            input_ids=None,
            input_video_embeds=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = self.config.use_return_dict if return_dict is None else return_dict  # noqa
        video_tokens = None if input_video_embeds is None else self.videomlp(input_video_embeds)
        return self.bert(input_ids, video_tokens, attention_mask=attention_mask, token_type_ids=token_type_ids,  # noqa
                         position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                         output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                         return_dict=return_dict)


class MMFusion(nn.Module):
    """a MMPT wrapper class for MMBert style models.
    TODO: move isolated mask to a subclass.
    """

    def __init__(self, max_video_len: int = 32, last_iso_layer: int = 12, num_hidden_video_layers: int = 6):
        super().__init__()
        self.model_name = "bert-base-uncased"
        transformer_config = AutoConfig.from_pretrained(self.model_name)
        self.hidden_size = transformer_config.hidden_size
        self.is_train = False
        # 0 means no iso; 1-12 means iso up to that layer.
        self.num_hidden_layers = transformer_config.num_hidden_layers
        self.last_iso_layer = last_iso_layer

        model_config = AutoConfig.from_pretrained(self.model_name)
        model_config.max_video_len = max_video_len
        # TODO: make each model a set of config class.
        if hasattr(model_config, "num_layers"):
            model_config.num_layers = num_hidden_video_layers
        else:
            model_config.num_hidden_layers = num_hidden_video_layers
        self.video_encoder = MMBertForEncoder.from_pretrained(self.model_name, config=model_config)
        # exact same NLP model from HuggingFace transformer.
        self.text_encoder = AutoConfig.from_pretrained("bert-base-uncased")

    def forward(
            self,
            caps,
            cmasks,
            vfeats,
            vmasks,
            **kwargs
    ):
        raise NotImplementedError(
            "Please derive MMFusion module."
        )

    def _mm_on_the_fly(
            self,
            cmasks,
            vmasks,
            attention_mask
    ):
        """helper function for mask, seg_ids and token_type_ids."""
        if attention_mask is None:
            attention_mask = self._mm_attention_mask(cmasks, vmasks)

        """
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        """
        token_type_ids = torch.cat([
            torch.zeros((vmasks.size(0), vmasks.size(1) + 2), dtype=torch.long, device=vmasks.device),
            torch.ones((cmasks.size(0), cmasks.size(1) - 2), dtype=torch.long, device=cmasks.device)], dim=1)
        return attention_mask, token_type_ids

    def _mm_attention_mask(self, cmasks, vmasks):
        assert cmasks.size(0) == vmasks.size(0), "{}, {}, {}, {}".format(
            str(cmasks.size()),
            str(vmasks.size()),
            str(cmasks.size(0)),
            str(vmasks.size(0)),
        )

        mm_mask = torch.cat([cmasks[:, :1], vmasks, cmasks[:, 1:]], dim=1)
        if self.last_iso_layer == 0:
            # hard attention mask.
            return mm_mask
        else:
            # a gpu iso mask; 0 : num_iso_layer is isolated;
            # num_iso_layer: are MM-fused.
            # make an iso layer
            batch_size = cmasks.size(0)
            iso_mask = self._make_iso_mask(batch_size, cmasks, vmasks)
            mm_mask = mm_mask[:, None, :].repeat(1, mm_mask.size(-1), 1)
            iso_mm_masks = []
            # hard attention mask.
            iso_mask = iso_mask[:, None, :, :].repeat(1, self.last_iso_layer, 1, 1)
            iso_mm_masks.append(iso_mask)
            if self.last_iso_layer < self.num_hidden_layers:
                mm_mask = mm_mask[:, None, :, :].repeat(1, self.num_hidden_layers - self.last_iso_layer, 1, 1)
                iso_mm_masks.append(mm_mask)
            iso_mm_masks = torch.cat(iso_mm_masks, dim=1)
            return iso_mm_masks

    def _make_iso_mask(self, batch_size, cmasks, vmasks):  # noqa
        cls_self_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, 1), dtype=torch.bool, device=cmasks.device),
                torch.zeros(
                    (batch_size, cmasks.size(1) + vmasks.size(1) - 1),
                    dtype=torch.bool, device=cmasks.device)
            ], dim=1)

        iso_video_mask = torch.cat(
            [
                # [CLS] is not used.
                torch.zeros(
                    (batch_size, 1), dtype=torch.bool, device=cmasks.device
                ),
                vmasks,
                # assume to be 1.
                cmasks[:, 1:2],
                # 2 means [CLS] + [SEP]
                torch.zeros(
                    (batch_size, cmasks.size(1) - 2),
                    dtype=torch.bool,
                    device=cmasks.device,
                ),
            ],
            dim=1,
        )
        iso_text_mask = torch.cat(
            [
                torch.zeros(
                    (batch_size, 2 + vmasks.size(1)),
                    dtype=torch.bool,
                    device=cmasks.device,
                ),  # [CLS] is not used.
                cmasks[:, 2:],  # assume to be 1.
            ],
            dim=1,
        )
        cls_self_mask = cls_self_mask[:, None, :]
        iso_video_mask = iso_video_mask[:, None, :].repeat(
            1, vmasks.size(1) + 1, 1)
        iso_text_mask = iso_text_mask[:, None, :].repeat(
            1, cmasks.size(1) - 2, 1)
        return torch.cat([cls_self_mask, iso_video_mask, iso_text_mask], dim=1)

    def _pooling_vt_layer(
            self,
            layered_sequence_output,
            cmasks,
            vmasks
    ):
        layer_idx = self.last_iso_layer \
            if self.last_iso_layer > 0 else self.num_hidden_layers
        hidden_state = layered_sequence_output[layer_idx]
        # also output pooled_video and pooled_text.
        batch_size = cmasks.size(0)
        # pool the modality.
        text_offset = vmasks.size(1) + 2  # [CLS] + [SEP]
        # video tokens + [SEP]
        video_outputs = hidden_state[:, 1:text_offset]
        video_attention_mask = torch.cat(
            [
                vmasks,
                torch.ones((batch_size, 1), dtype=torch.bool, device=vmasks.device),
            ],
            dim=1,
        )
        assert video_outputs.size(1) == video_attention_mask.size(1)
        pooled_video = (torch.sum(video_outputs * video_attention_mask.unsqueeze(-1), dim=1)
                        / video_attention_mask.sum(1, keepdim=True))
        # pooled_video = torch.mean(video_outputs[0], dim=1)

        # text tokens + [SEP]
        text_attention_mask = cmasks[:, 2:]
        text_outputs = hidden_state[:, text_offset:]
        assert text_outputs.size(1) == text_attention_mask.size(1)
        pooled_text = torch.sum(
            text_outputs * text_attention_mask.unsqueeze(-1), dim=1
        ) / text_attention_mask.sum(1, keepdim=True)
        return pooled_video, pooled_text


class MMFusionSeparate(MMFusion):
    def forward(
            self,
            caps,
            cmasks,
            vfeats,
            vmasks,
            attention_mask=None,
            video_label=None,
            text_label=None,
            output_hidden_states=False,
            **kwargs
    ):
        pooled_video = self.forward_video(
            vfeats,
            vmasks,
            caps,
            cmasks,
            output_hidden_states
        )

        pooled_text = self.forward_text(
            caps,
            cmasks,
            output_hidden_states
        )

        return {"pooled_video": pooled_video, "pooled_text": pooled_text}

    def forward_video(
            self,
            vfeats,
            vmasks,
            caps,
            cmasks,
            output_hidden_states=False,
            **kwargs  # noqa
    ):
        input_ids = caps[:, :2]

        attention_mask = torch.cat([cmasks[:, :1], vmasks, cmasks[:, 1:2]], dim=1)

        token_type_ids = torch.zeros(
            (vmasks.size(0), vmasks.size(1) + 2),
            dtype=torch.long,
            device=vmasks.device)

        outputs = self.video_encoder(
            input_ids=input_ids,
            input_video_embeds=vfeats,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        video_outputs = outputs[0]

        if output_hidden_states:
            return video_outputs

        batch_size = cmasks.size(0)

        video_attention_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=vmasks.device),
                                          vmasks, torch.ones((batch_size, 1), dtype=torch.bool, device=vmasks.device)],
                                         dim=1)

        assert video_outputs.size(1) == video_attention_mask.size(1)
        video_attention_mask = video_attention_mask.type(video_outputs.dtype) / video_attention_mask.sum(1,
                                                                                                         keepdim=True)
        return torch.bmm(video_outputs.transpose(2, 1), video_attention_mask.unsqueeze(2)).squeeze(-1)

    def forward_text(
            self,
            caps,
            cmasks,
            output_hidden_states=False,
            **kwargs  # noqa
    ):
        input_ids = torch.cat([
            caps[:, :1], caps[:, 2:],
        ], dim=1)

        attention_mask = torch.cat([
            cmasks[:, :1],
            cmasks[:, 2:]
        ], dim=1)
        # different from sharing, we use all-0 type.
        token_type_ids = torch.zeros(
            (cmasks.size(0), cmasks.size(1) - 1),
            dtype=torch.long,
            device=cmasks.device)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        text_outputs = outputs[0]

        if output_hidden_states:
            return text_outputs

        batch_size = caps.size(0)
        # text tokens + [SEP]
        text_attention_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=cmasks.device),
                                         cmasks[:, 2:]], dim=1)

        assert text_outputs.size(1) == text_attention_mask.size(1)
        text_attention_mask = text_attention_mask.type(text_outputs.dtype) / text_attention_mask.sum(1, keepdim=True)
        return torch.bmm(text_outputs.transpose(2, 1), text_attention_mask.unsqueeze(2)).squeeze(-1)
