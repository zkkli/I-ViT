import os
import math
import re
import warnings
from itertools import repeat
import collections.abc
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from .layers_quant import PatchEmbed, Mlp, DropPath, trunc_normal_
from .quantization_utils import QuantLinear, QuantAct, QuantConv2d, IntLayerNorm, IntSoftmax, IntGELU, QuantMatMul
from .utils import load_weights_from_npz


__all__ = ['deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
           'vit_base_patch16_224', 'vit_large_patch16_224']


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = QuantLinear(
            dim,
            dim * 3,
            bias=qkv_bias
        )
        self.qact1 = QuantAct()
        self.qact_attn1 = QuantAct()
        self.qact2 = QuantAct()
        self.proj = QuantLinear(
            dim,
            dim
        )
        self.qact3 = QuantAct(16)
        self.qact_softmax = QuantAct()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.int_softmax = IntSoftmax(16)

        self.matmul_1 = QuantMatMul()
        self.matmul_2 = QuantMatMul()

    def forward(self, x, act_scaling_factor):
        B, N, C = x.shape
        x, act_scaling_factor = self.qkv(x, act_scaling_factor)
        x, act_scaling_factor_1 = self.qact1(x, act_scaling_factor)
        qkv = x.reshape(B, N, 3, self.num_heads, C //
                        self.num_heads).permute(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        attn, act_scaling_factor = self.matmul_1(q, act_scaling_factor_1,
                                                 k.transpose(-2, -1), act_scaling_factor_1)
        attn = attn * self.scale
        act_scaling_factor = act_scaling_factor * self.scale
        attn, act_scaling_factor = self.qact_attn1(attn, act_scaling_factor)

        attn, act_scaling_factor = self.int_softmax(attn, act_scaling_factor)

        attn = self.attn_drop(attn)
        x, act_scaling_factor = self.matmul_2(attn, act_scaling_factor,
                                              v, act_scaling_factor_1)
        x = x.transpose(1, 2).reshape(B, N, C)

        x, act_scaling_factor = self.qact2(x, act_scaling_factor)
        x, act_scaling_factor = self.proj(x, act_scaling_factor)
        x, act_scaling_factor = self.qact3(x, act_scaling_factor)
        x = self.proj_drop(x)

        return x, act_scaling_factor


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.qact1 = QuantAct()
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.qact2 = QuantAct(16)
        self.norm2 = norm_layer(dim)
        self.qact3 = QuantAct()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
        self.qact4 = QuantAct(16)

    def forward(self, x_1, act_scaling_factor_1):
        x, act_scaling_factor = self.norm1(x_1, act_scaling_factor_1)
        x, act_scaling_factor = self.qact1(x, act_scaling_factor)
        x, act_scaling_factor = self.attn(x, act_scaling_factor)
        x = self.drop_path(x)
        x_2, act_scaling_factor_2 = self.qact2(x, act_scaling_factor, x_1, act_scaling_factor_1)

        x, act_scaling_factor = self.norm2(x_2, act_scaling_factor_2)
        x, act_scaling_factor = self.qact3(x, act_scaling_factor)
        x, act_scaling_factor = self.mlp(x, act_scaling_factor)
        x = self.drop_path(x)
        x, act_scaling_factor = self.qact4(x, act_scaling_factor, x_2, act_scaling_factor_2)

        return x, act_scaling_factor


class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            representation_size=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.qact_input = QuantAct()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.qact_pos = QuantAct(16)
        self.qact1 = QuantAct(16)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    act_layer=IntGELU,
                    norm_layer=norm_layer
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.qact2 = QuantAct()

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = (
            QuantLinear(
                self.num_features,
                num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.act_out = QuantAct()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]

        x, act_scaling_factor = self.qact_input(x)
        x, act_scaling_factor = self.patch_embed(x, act_scaling_factor)
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  # share scaling_factor

        x_pos, act_scaling_factor_pos = self.qact_pos(self.pos_embed)
        x, act_scaling_factor = self.qact1(x, act_scaling_factor, x_pos, act_scaling_factor_pos)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x, act_scaling_factor = blk(x, act_scaling_factor)

        x, act_scaling_factor = self.norm(x, act_scaling_factor)
        x = x[:, 0]
        x, act_scaling_factor = self.qact2(x, act_scaling_factor)
        x = self.pre_logits(x)

        return x, act_scaling_factor

    def forward(self, x):
        x, act_scaling_factor = self.forward_features(x)
        x, act_scaling_factor = self.head(x, act_scaling_factor)
        #x, _ = self.act_out(x, act_scaling_factor)
        return x


def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(IntLayerNorm, eps=1e-6),
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(IntLayerNorm, eps=1e-6),
        **kwargs
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(IntLayerNorm, eps=1e-6),
        **kwargs
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(IntLayerNorm, eps=1e-6),
        **kwargs
    )
    if pretrained:
        url = "https://storage.googleapis.com/vit_models/augreg/" + \
            "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"

        load_weights_from_npz(model, url, check_hash=True)
    return model


def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(IntLayerNorm, eps=1e-6),
        **kwargs
    )
    if pretrained:
        url = "https://storage.googleapis.com/vit_models/augreg/" + \
            "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"

        load_weights_from_npz(model, url, check_hash=True)
    return model
