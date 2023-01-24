# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copied from https://github.com/facebookresearch/SLIP/tree/c6faf5d

import gzip
import html
from collections import OrderedDict
from functools import lru_cache
from typing import Iterable, Iterator

import ftfy
import numpy as np
import regex as re
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from cached_path import cached_path
from torch import autograd


@lru_cache()
def default_bpe():
    return cached_path("https://github.com/facebookresearch/SLIP/raw/main/bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer:
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join(self.decoder[token] for token in tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def __call__(self, texts, context_length=77):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            tokens = tokens[:context_length]
            result[i, :len(tokens)] = torch.tensor(tokens)

        if len(result) == 1:
            return result[0]
        return result


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_list = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    return [torch.cat(tensor_all) for tensor_all in tensor_list]


class GatherLayer(autograd.Function):  # noqa
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):  # noqa
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    return tensors if get_world_size() == 1 else [torch.cat(GatherLayer.apply(tensor)) for tensor in tensors]


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        image_embed = outputs['image_embed']
        text_embed = outputs['text_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = all_gather_batch([image_embed, text_embed])

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + F.cross_entropy(logits_per_text, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'clip_loss': loss, 'clip_acc': acc}


class SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        q_a = outputs['aug1_embed']
        q_b = outputs['aug2_embed']

        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)

        local_batch_size = q_a.size(0)

        k_a, k_b = all_gather_batch_with_grad([q_a, q_b])

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=q_a.device
            )
            total_batch_size = local_batch_size * get_world_size()
            self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ssl_loss': loss, 'ssl_acc': acc}


class SLIPLoss(nn.Module):
    def __init__(self, ssl_loss, ssl_scale):
        super().__init__()
        self.clip_loss = CLIPLoss()
        self.ssl_loss = ssl_loss
        self.ssl_scale = ssl_scale

    def forward(self, outputs):
        clip_loss_dict = self.clip_loss(outputs)
        clip_loss = clip_loss_dict['clip_loss']
        clip_acc = clip_loss_dict['clip_acc']

        ssl_loss_dict = self.ssl_loss(outputs)
        ssl_loss = ssl_loss_dict['ssl_loss']
        ssl_acc = ssl_loss_dict['ssl_acc']

        return {'loss': clip_loss + self.ssl_scale * ssl_loss,
                'clip_loss': clip_loss,
                'clip_acc': clip_acc,
                'ssl_loss': ssl_loss,
                'ssl_acc': ssl_acc}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):  # noqa
        return x * torch.sigmoid(1.702 * x)  # noqa


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        x = self.visual(image)
        x = x @ self.image_projection

        return x

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp()}


def _build_mlp(in_dim, mlp_dim, out_dim):
    return nn.Sequential(OrderedDict([
        ("layer1", nn.Linear(in_dim, mlp_dim)),
        ("bn1", nn.SyncBatchNorm(mlp_dim)),
        ("relu1", nn.ReLU(inplace=True)),
        ("layer2", nn.Linear(mlp_dim, mlp_dim)),
        ("bn2", nn.SyncBatchNorm(mlp_dim)),
        ("relu2", nn.ReLU(inplace=True)),
        ("layer3", nn.Linear(mlp_dim, out_dim)),
    ]))


class SIMCLR(nn.Module):
    def __init__(self,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # ssl
                 ssl_mlp_dim: int,
                 ssl_emb_dim: int,
                 **kwargs,  # noqa
                 ):
        super().__init__()
        self.vision_width = vision_width
        self.visual = vision_model
        self.image_mlp = _build_mlp(in_dim=vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)

    def encode_image(self, image):
        return self.visual(image)

    def forward(self, aug1, aug2):
        h1 = self.visual(aug1)
        h2 = self.visual(aug2)

        aug1_embed = self.image_mlp(h1)
        aug2_embed = self.image_mlp(h2)

        return {'aug1_embed': aug1_embed, 'aug2_embed': aug2_embed}


class SLIP(CLIP):
    def __init__(self, ssl_mlp_dim: int, ssl_emb_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.image_mlp = _build_mlp(in_dim=self.vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)

    def forward(self, image, text, aug1, aug2):  # noqa
        aug1_embed = self.image_mlp(self.visual(aug1))
        aug2_embed = self.image_mlp(self.visual(aug2))

        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {'image_embed': image_embed, 'text_embed': text_embed, 'logit_scale': self.logit_scale.exp(),
                'aug1_embed': aug1_embed, 'aug2_embed': aug2_embed}


def get_loss(model, ssl_temp, ssl_scale):
    if model.startswith('SLIP'):
        ssl_loss = SIMCLRLoss(temperature=ssl_temp)
        return SLIPLoss(ssl_loss, ssl_scale)
    if model.startswith('CLIP'):
        return CLIPLoss()
    if model.startswith('SIMCLR'):
        return SIMCLRLoss(temperature=ssl_temp)


def get_metric_names(model):
    if model.startswith('SLIP'):
        return ['loss', 'clip_loss', 'ssl_loss', 'clip_acc', 'ssl_acc']
    elif model.startswith('CLIP'):
        return ['loss', 'clip_loss', 'clip_acc']
    else:
        return ['loss', 'ssl_loss', 'ssl_acc']


@timm.models.registry.register_model
def vit_small_mocov3_patch16_224(**kwargs):
    return timm.models.vision_transformer._create_vision_transformer("vit_small_patch16_224", patch_size=16,
                                                                     embed_dim=384, depth=12, num_heads=12, **kwargs)


def CLIP_VITS16(**kwargs):
    vision_model = timm.create_model('vit_small_mocov3_patch16_224', num_classes=0)
    model = CLIP(embed_dim=512, vision_width=384, vision_model=vision_model, context_length=77, vocab_size=49408,
                 transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def SIMCLR_VITS16(**kwargs):
    vision_model = timm.create_model('vit_small_mocov3_patch16_224', num_classes=0)
    model = SIMCLR(vision_width=384, vision_model=vision_model, **kwargs)

    return model


def SLIP_VITS16(**kwargs):
    vision_model = timm.create_model('vit_small_mocov3_patch16_224', num_classes=0)
    model = SLIP(embed_dim=512, vision_width=384, vision_model=vision_model, context_length=77, vocab_size=49408,
                 transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def CLIP_VITB16(**kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    model = CLIP(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
                 transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def SIMCLR_VITB16(**kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    model = SIMCLR(vision_width=768, vision_model=vision_model, **kwargs)

    return model


def SLIP_VITB16(**kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    model = SLIP(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
                 transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def CLIP_VITL16(**kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0)
    model = CLIP(embed_dim=512, vision_width=1024, vision_model=vision_model, context_length=77, vocab_size=49408,
                 transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model


def SIMCLR_VITL16(**kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0)
    model = SIMCLR(vision_width=1024, vision_model=vision_model, **kwargs)

    return model


def SLIP_VITL16(**kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0)
    model = SLIP(embed_dim=512, vision_width=1024, vision_model=vision_model, context_length=77, vocab_size=49408,
                 transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model
