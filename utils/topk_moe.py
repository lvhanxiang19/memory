import torch
from torch import nn, einsum
import torch.nn.functional as F
from collections import namedtuple
from typing import Tuple
from functools import partial
import einx
from einops import rearrange, repeat, reduce, pack, unpack

from colt5_attention import topk as maybe_differentiable_topk

from .fast_softmoe_layer import MultiExpertLayer

# =============== Helper functions ===============

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d() if callable(d) else d

def cast_tuple(el, len=1):
    return el if isinstance(el, tuple) else ((el,) * len)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]

MixtureOfExpertsReturn = namedtuple('MixtureOfExpertsReturn', [
    'outputs',
    'total_aux_loss',
    'balance_loss',
    'router_z_loss'
])

MIN_EXPERT_CAPACITY = 4

class TopNGating(nn.Module):

    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        top_n = 2,
        threshold_train = 0.2,
        threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True
    ):
        super().__init__()
        self.eps = eps
        self.num_gates = num_gates
        self.to_gates = nn.Linear(dim, num_gates, bias = False)

        self.differentiable_topk = differentiable_topk

        self.topk = partial(
            maybe_differentiable_topk,
            non_differentiable = not differentiable_topk,
            fused = differentiable_topk_fused # use triton fused coordinate descent if possible by default
        )

        assert top_n >= 2, 'must be 2 or more experts'
        self.top_n = top_n
        top_n_minus_1 = top_n - 1

        threshold_train = cast_tuple(threshold_train, top_n_minus_1)
        threshold_eval = cast_tuple(threshold_eval, top_n_minus_1)

        assert len(threshold_train) == len(threshold_eval) == top_n_minus_1

        self.register_buffer('threshold_train', torch.tensor([eps, *threshold_train]))
        self.register_buffer('threshold_eval', torch.tensor([eps, *threshold_eval]))

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval        

        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.register_buffer('zero', torch.zeros((1,)), persistent = False)

    def forward(
        self,
        x,
        noise_gates = False,
        noise_mult = 1.
    ):
        """
        einstein notation:

        b - batch
        n - sequence
        e - experts
        k - top-n experts
        c - capacity
        """

        *_, b, group_size, dim, dtype, top_n, num_gates, eps = *x.shape, x.dtype, self.top_n, self.num_gates, self.eps

        # threshold, capacity depending on training or eval

        suffix = 'train' if self.training else 'eval'

        threshold = getattr(self, f'threshold_{suffix}')
        capacity_factor = getattr(self, f'capacity_factor_{suffix}')

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes

        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # gate logits and gates

        gate_logits = self.to_gates(x)

        maybe_noised_gate_logits = gate_logits

        if noise_gates:
            noise = gumbel_noise(maybe_noised_gate_logits)
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise * noise_mult

        raw_gates = maybe_noised_gate_logits.softmax(dim = -1)

        # find top N experts per position

        topk_return = self.topk(raw_gates, k = top_n)

        gate_indices = topk_return.indices

        if self.differentiable_topk:
            # allow for differentiable topk using coordinate descent
            # used successfully for routing from CoLT5 paper https://github.com/lucidrains/CoLT5-attention

            gates = topk_return.coor_descent_values
        else:
            gates = topk_return.values

        # move the top-n dimension to be first

        gates = rearrange(gates, '... k -> k ...')
        gate_indices = rearrange(gate_indices, '... k -> k ...')##从b，s，k变成k，s，b可以认为是一共有k个indices，具有顺序，第一个就是最先激活的专家

        # masks

        one_hot_gate_indices = F.one_hot(gate_indices, num_gates)###从k，b，s变成k，b，s，e。e为专家数量
        mask = one_hot_gate_indices.float()

        mask_1 = mask[0] # needed for balancing loss 一个batch中的每一个token最先激活的专家的编号

        # normalize top-n gate scores

        denom = reduce(gates, 'k ... -> 1 ...', 'sum').clamp(min = eps) 
        gates = gates / denom  ##将topk的概率加权平均

        # best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
        # optimal threshold was ~ 0.2
        # generalized to more than 2 experts

        probs = torch.zeros_like(gates).uniform_(0., 1.)

        should_route = probs < einx.divide('k b n, k -> k b n', gates, threshold.clamp(min = eps))##比较概率，如果该概率小于预设概率，则不激活，为false

        # tokens should always be routed to first expert
        # threshold for first expert already set to very small number, but just in case

        should_route[0, ...] = True##保证第一个专家要激活

        mask *= rearrange(should_route.float(), '... -> ... 1') ##mask的维度是k，b，s，e should——route的维度在未添加之前是k，b，s，添加之后是k，b，s，1 这样的目的是如果topk的专家无法激活时的indice就是[0，0---,0]

        mask_cumsum = cumsum_exclusive(mask, dim = -2) # along sequence dimension 

        # compute assignment to experts - (batch, seq, experts)

        # This is the position within the expert's mini-batch for this sequence

        positions = []
        prev_expert_count = 0.

        for n in range(self.top_n):
            position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]  #mask[n]是指第n个排序的专家的编号，mask_cumsum[n]是指该专家是否被激活，pre是指该专家之前被激活的次数

            # Remove the elements that don't fit. (batch, sequence, experts)
            mask[n] *= (position_in_expert < expert_capacity_f).float()    ##如果该专家的激活次数超过了最大容量，则不激活该专家

            # How many examples in this sequence go to this expert - needed for the next iteration as offset
            prev_expert_count = reduce(mask[n], '... n e -> ... 1 e', 'sum') + prev_expert_count

            # (batch, sequence)
            position_in_expert = reduce(position_in_expert, '... n e -> ... n', 'sum')
            positions.append(position_in_expert)

        positions = torch.stack(positions)

        # (k, batch, sequence) - mostly ones, but zeros where something didn't fit
        mask_flat = reduce(mask, '... n e -> ... n', 'sum')

        # (k, batch, sequence) - weighted assignment
        # following https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L1903
        gates = gates * mask_flat

        # (batch, sequence, experts, expert_capacity)

        combine_tensor = einx.multiply(
            'k b n, k b n, k b n e, k b n c -> k b n e c',
            gates,
            mask_flat,
            one_hot_gate_indices,
            safe_one_hot(positions.long(), expert_capacity)
        )

        combine_tensor = reduce(combine_tensor, 'k b n e c -> b n e c', 'sum')

        # dispatch tensor

        dispatch_tensor = combine_tensor.bool().type(dtype)

        if self.straight_through_dispatch_tensor:
            dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()

        # balance losses - (batch, experts)
        # We want to equalize the fraction of the batch assigned to each expert

        if self.training:
            density_1 = reduce(mask_1, '... n e -> ... e', 'mean')
            density_1_proxy = reduce(raw_gates, '... n e -> ... e', 'mean') # Something continuous that is correlated with what we want to equalize.

            balance_loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)
        else:
            balance_loss = self.zero

        # calculate the router z-loss proposed in paper

        if self.training:
            router_z_loss = torch.logsumexp(gate_logits, dim = -1)
            router_z_loss = torch.square(router_z_loss)            
            router_z_loss = router_z_loss.mean()
        else:
            router_z_loss = self.zero

        return dispatch_tensor, combine_tensor, balance_loss, router_z_loss

# plain mixture of experts

class MoE(nn.Module):

    def __init__(self,
        dim,
        num_experts = 16,
        expert_hidden_mult = 4,
        threshold_train = 0.2,
        threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        gating_top_n = 2,
        balance_loss_coef = 1e-2,
        router_z_loss_coef = 1e-3,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True,
        is_distributed = None,
        allow_var_seq_len = False
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        self.gate = TopNGating(
            dim,
            top_n = gating_top_n,
            num_gates = num_experts,
            straight_through_dispatch_tensor = straight_through_dispatch_tensor,
            differentiable_topk = differentiable_topk,
            threshold_train = threshold_train,
            threshold_eval = threshold_eval,
            capacity_factor_train = capacity_factor_train,
            capacity_factor_eval = capacity_factor_eval
        )

        self.experts = MultiExpertLayer(
            in_dim=dim, 
            hidden_dim=dim*expert_hidden_mult, 
            num_experts=num_experts, 
        )

        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

    def forward(
        self,
        x,
        noise_gates = False,
        noise_mult = 1.
    ):
        dispatch_tensor, combine_tensor, balance_loss, router_z_loss = self.gate(x, noise_gates = noise_gates, noise_mult = noise_mult)

        # dispatch

        expert_inputs = einsum('b n d, b n e c -> b e c d', x, dispatch_tensor)

        # feed the expert inputs through the experts.

        expert_outputs = self.experts(expert_inputs)

        # combine

        output = einsum('b e c d, b n e c -> b n d', expert_outputs, combine_tensor)

        # losses

        weighted_balance_loss = balance_loss * self.balance_loss_coef
        weighted_router_z_loss = router_z_loss * self.router_z_loss_coef

        # combine the losses

        total_aux_loss = weighted_balance_loss + weighted_router_z_loss

        return output, total_aux_loss