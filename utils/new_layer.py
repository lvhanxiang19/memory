import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked,GluMlp,SwiGLU,\
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations
from .memory import HashingMemory
from .topk_moe import MoE
from PEER_pytorch import PEER
from .ultra import UltraMemory



class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class newffn(nn.Module):
    def __init__(
            self,
            dim,
            mem_dim:int,
            mem_k_dim:int,
            mem_k_num:int,
            mem_head:int,
            mem_knn:int,

            value_lr:float,
            num_experts:int,
            top_n:int,
            expert_hidden_mult:int
        ) :
        super().__init__()
        self.MoE=MoE(dim=dim,num_experts=num_experts,gating_top_n=top_n,expert_hidden_mult=expert_hidden_mult)
        '''self.memory=HashingMemory(
                input_dim = mem_dim,
                output_dim = mem_dim,
                mem_n_keys = mem_k_num,
                mem_heads= mem_head,
                mem_knn= mem_knn,
                mem_share_values= False,
                mem_k_dim= mem_k_dim,
                mem_v_dim = -1,
                swilu_projection= False,
                value_fixed_lr= value_lr,
                peer_variant = False,
            )'''
        self.mlp_main=Mlp(
            in_features=dim,
            hidden_features=int(dim * 2),
            out_features=dim)
    def forward(self,x):
        x_t,loss=x
        x_1,loss_t=self.MoE(x_t)
        #x_2=self.memory(x_t)
        x_2=self.mlp_main(x_t)
        return (x_1+x_2,loss+loss_t)



class newBlock(nn.Module):
    def __init__(
            self,
            ##memory settings
            mem_dim,
            mem_k_dim,
            mem_k_num,
            mem_head,
            mem_knn,
            value_lr,
            
            ##MoE settings
            num_experts,
            top_n,
            expert_hidden_mult,

            ##origin settings
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            is_alt:bool=False
            
    )  -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.alt=is_alt
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        if is_alt:
            self.mlp=newffn(
                dim=dim,
                mem_dim=mem_dim,
                mem_k_dim=mem_k_dim,
                mem_k_num=mem_k_num,
                mem_head=mem_head,
                mem_knn=mem_knn,
                value_lr=value_lr,
                num_experts=num_experts,
                top_n=top_n,
                expert_hidden_mult=expert_hidden_mult
            )
        else:
         self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
           
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x) :
        x,loss=x
        loss_t=0
        if self.alt:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            y ,loss_t = self.mlp((self.norm2(x),loss))
            x = x + self.drop_path2(self.ls2(y))
        else:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        z=(x,loss+loss_t)
        return z