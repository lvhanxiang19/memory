import torch
import torch.nn as nn
import torch.nn.functional as F
import einx
import math
import warnings
from torch import Tensor
from einops import rearrange, reduce, einsum
from functools import partial
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from timm.layers import trunc_normal_

from colt5_attention import topk as maybe_differentiable_topk


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d() if callable(d) else d

def cast_tuple(el, len=1):
    return el if isinstance(el, tuple) else ((el,) * len)

def log(t, eps=1e-10):
    return t.clamp(min=eps).log()

def gumbel_like(t, eps=1e-10):
    noise = torch.rand_like(t)
    return -log(-log(noise, eps), eps)

def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]

# def sinkhorn(
#     logits: torch.Tensor,
#     n_iters: int = 6,
#     temperature: float = 1.0,
# ) -> torch.Tensor:
#     t = logits / temperature

#     for _ in range(n_iters):
#         t = t - t.logsumexp(dim=-2, keepdim=True)
#         t = t - t.logsumexp(dim=-1, keepdim=True)
#     return torch.exp(t)

def sinkhorn(
    logits: torch.Tensor,
    n_iters: int = 8,
    eps: float = 1e-4,
    tau: float = 1.0
    ):
    t = logits / tau
    k = t.size(-2)/t.size(-1)
    for _ in range(n_iters):
        t = t - t.logsumexp(dim=-2, keepdim=True)
        t = t - t.logsumexp(dim=-1, keepdim=True)

        if _ > 1:
            P = t.exp()
            if (P.sum(dim=-1).sub_(1).abs().max() < eps and
                P.sum(dim=-2).sub_(k).abs().max() < eps):
                break
        
    return t.exp()
    
class MeanExpander(nn.Module):
    def __init__(self, slots_per_expert, learnable=True, dim=64):
        super().__init__()
        self.slots_per_expert = slots_per_expert
        self.learnable = learnable
        if self.learnable:
            self.proj = nn.Linear(dim, slots_per_expert)
            #  self.proj = EnhancedKroneckerLinear(dim, slots_per_expert)
    
    def forward(self, x):
        # 计算均值并保持维度，然后扩展到指定大小
        if self.learnable:
            return self.proj(x)
        return torch.mean(x, dim=-1, keepdim=True).repeat(1, 1, self.slots_per_expert)

def l2norm(x, dim=-1, eps=1e-6):
    norm = torch.sqrt(torch.sum(x**2, dim=dim, keepdim=True))
    return x * (1 / (norm + eps))


class L2Norm(nn.Module):
    def __init__(self, dim=-1, eps=1e-6, fp32_norm=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.fp32_norm = fp32_norm

    def forward(self, x):
        if self.fp32_norm:      # ⇽ 临时升级到 fp32
            dtype = x.dtype
            x_fp32 = x.float()
            return F.normalize(x_fp32, p=2, dim=self.dim, eps=self.eps).to(dtype)
        else:
            return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

def get_ratio_str(subset_ratio):
    return f'{subset_ratio:.2f}'.replace('.', 'p')

def closest_factors(n, larger_first=False):
    sqrt_n = int(math.sqrt(n))
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            if larger_first:
                return n // i, i
            else:
                return i, n // i
    return 1, n  # 默认返回 (1, n) 如果没有找到其他因子


class KroneckerLinear(nn.Module):
    def __init__(self, in_features, out_features, 
                 in_factors=None, out_factors=None, 
                 bias=True, zero_init=False, orthogonal_init=False, 
                 num_heads=1):

        super(KroneckerLinear, self).__init__()
        
        # Factor input
        if in_factors is not None:
            assert in_factors[0] * in_factors[1] == in_features, "in_factors do not multiply to in_features"
            self.in_features1, self.in_features2 = in_factors
        else:
            self.in_features1, self.in_features2 = closest_factors(in_features)
        
        # Factor output
        if out_factors is not None:
            assert out_factors[0] * out_factors[1] == out_features, "out_factors do not multiply to out_features"
            self.out_features1, self.out_features2 = out_factors
        else:
            self.out_features1, self.out_features2 = closest_factors(out_features)
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        
        # Warn if factors are trivial
        if self.in_features1 == 1 or self.in_features2 == 1:
            warnings.warn(f"in_features({in_features}) factorized as ({self.in_features1}, {self.in_features2}). Consider specifying factors manually for better performance.")
        if self.out_features1 == 1 or self.out_features2 == 1:
            warnings.warn(f"out_features({out_features}) factorized as ({self.out_features1}, {self.out_features2}). Consider specifying factors manually for better performance.")
        
        # Initialize parameters: weight1: (num_heads, out_f1, in_f1), weight2: (num_heads, out_f2, in_f2)
        self.weight1 = nn.Parameter(torch.empty(self.num_heads, self.in_features1, self.out_features1))
        self.weight2 = nn.Parameter(torch.empty(self.num_heads, self.in_features2, self.out_features2))
        
        for h in range(self.num_heads):
            if orthogonal_init:
                nn.init.orthogonal_(self.weight1[h])
                nn.init.orthogonal_(self.weight2[h])
            else:
                nn.init.kaiming_uniform_(self.weight1[h], a=math.sqrt(5))
                if zero_init:
                    nn.init.zeros_(self.weight2[h])
                else:
                    nn.init.kaiming_uniform_(self.weight2[h], a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
            if zero_init:
                nn.init.zeros_(self.bias)
            else:
                bound = 1 / math.sqrt(in_features)
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """
        x: shape could be (batch_size, n, in_features) or (batch_size*n, in_features) or (in_features,).
        
        Returns:
            Tensor of shape corresponding to input, but out_features dimension is the same as single-head:
            - If (batch_size, n, in_features) -> (batch_size, n, out_features)
            - If (batch_size*n, in_features) -> (batch_size*n, out_features)
            - If (in_features,) -> (out_features,)
        """
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, n, in_features = x.size()
            assert in_features == self.in_features, f"Expected {self.in_features}, got {in_features}"
            x = x.reshape(batch_size * n, in_features)
        elif x.dim() == 2:
            # (n, in_features)
            n, in_features = x.size()
            assert in_features == self.in_features, f"Expected {self.in_features}, got {in_features}"
        elif x.dim() == 1:
            # (in_features,)
            in_features = x.size(0)
            assert in_features == self.in_features, f"Expected {self.in_features}, got {in_features}"
            x = x.unsqueeze(0)
        
        batch_n = x.size(0)

        x = x.view(batch_n, self.in_features1, self.in_features2) 

        x = x.unsqueeze(0).expand(self.num_heads, -1, -1, -1) 
        
        x = einsum(x, self.weight2, 'h b i1 i2, h i2 o2 ->h b i1 o2')
        x = einsum(x, self.weight1, 'h b i1 o2, h i1 o1 ->h b o1 o2')

        x = x.sum(dim=0)
        
        # 4. Reshape to (batch_n, out_features)
        x = x.reshape(batch_n, self.out_features)
        
        # 5. Add bias summed over heads if needed
        if self.bias is not None:
            x += self.bias
        
        # Reshape back if needed
        if len(original_shape) == 3:
            x = x.view(batch_size, n, self.out_features)
        elif len(original_shape) == 1:
            x = x.squeeze(0)
        
        return x


class EnhancedKroneckerLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, bias=True, zero_init=False, orthogonal_init=False, num_heads=1):
        super().__init__()
        self.kron = KroneckerLinear(
            in_features, out_features, bias=bias, 
            zero_init=zero_init, orthogonal_init=orthogonal_init, num_heads=num_heads
        )
                 
        self.low_rank = nn.Sequential(
            nn.Linear(in_features, rank, bias=False),
            nn.Linear(rank, out_features, bias=True),
        )

        nn.init.kaiming_uniform_(self.low_rank[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.low_rank[1].bias)  # 偏置初始化为0
        
        if zero_init:
            nn.init.zeros_(self.low_rank[1].weight)
        else:
            nn.init.kaiming_uniform_(self.low_rank[1].weight, a=math.sqrt(5))
        
        self.low_rank[0].skip_init = True
        self.low_rank[1].skip_init = True

    def forward(self, x):
        return self.kron(x) + self.low_rank(x)


class FixedLearnablePad(nn.Module):
    def __init__(self, input_len, dim):
        super().__init__()
        h = int(math.ceil(math.sqrt(input_len)))
        k = h * h
        pad_len = k - input_len
        self.h = h
        self.w = h
        self.pad_len = pad_len
        self.input_len = input_len

        if pad_len > 0:
            self.pad_tokens = nn.Parameter(torch.randn(pad_len, dim))
        else:
            self.pad_tokens = None

    def forward(self, x):
        # x: (b, n, d)
        if self.pad_len == 0:
            return x, self.h, self.w
        b = x.size(0)
        pad = self.pad_tokens.unsqueeze(0).expand(b, -1, -1)  # (b, pad_len, d)
        x_padded = torch.cat([x, pad], dim=1)
        return x_padded, self.h, self.w

    def unpad(self, x):
        x = x[:, :self.input_len, :]  # (b, n, d)
        return x

class DynamicPad(nn.Module):
    def __init__(self):
        super().__init__()
        self.original_len = None  # Initialize state

    def forward(self, x):
        b, n, d = x.shape
        self.original_len = n

        h = int(math.ceil(math.sqrt(n)))
        padded_len = h * h
        pad_len = padded_len - n

        if pad_len == 0:
            return x, h, h

        padded_x = F.pad(x, (0, 0, 0, pad_len))
        
        return padded_x, h, h

    def unpad(self, x):
        if self.original_len is None:
            raise RuntimeError("You must call forward before unpad.")
            
        return x[:, :self.original_len, :]

class OffsetScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        # nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = x * self.gamma + self.beta
        return out

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

MIN_EXPERT_CAPACITY = 4

class TokenChoiceGating(nn.Module):

    def __init__(
        self,
        subset_experts_mapping: Dict[float, int] = {1.0: 1},
        add_noise: bool = False,
        noise_mult: float = 1.0,
        softmax_temp: List[float] = [0.5, 0.5],
        #
        eps = 1e-9,
        top_n = 2,
        threshold_train = 0.2,
        threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.0,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True,
        #
        use_sinkhorn = True,
        sinkhorn_iters: int = 6,
        **kwargs: Any,
    ):
        super().__init__()
        self.eps = eps
        self.num_experts = sum(subset_experts_mapping.values())

        self.add_noise = add_noise
        self.noise_mult = noise_mult
        self.softmax_temp = nn.Parameter(torch.log(torch.tensor(softmax_temp[-1], dtype=torch.float)))

        self.differentiable_topk = differentiable_topk

        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iters  = sinkhorn_iters

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

    def forward(self, key, query):
        """
        einstein notation:

        b - batch
        n - sequence
        e - experts
        k - top-n experts
        c - capacity
        """

        *_, b, group_size, dim, dtype, top_n, num_experts, eps = *key.shape, key.dtype, self.top_n, self.num_experts, self.eps

        # threshold, capacity depending on training or eval

        suffix = 'train' if self.training else 'eval'

        threshold = getattr(self, f'threshold_{suffix}')
        capacity_factor = getattr(self, f'capacity_factor_{suffix}')

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes

        expert_capacity = min(group_size, int((group_size * capacity_factor) / self.num_experts))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # gate logits and gates

        gate_logits = torch.einsum("b n d, e d -> b n e", key, query)

        maybe_noised_gate_logits = gate_logits

        if self.add_noise and self.training:
            noise_std = (1.0 / self.num_experts) * self.noise_mult
            noise = torch.randn_like(maybe_noised_gate_logits) * noise_std
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise

        maybe_noised_gate_logits = maybe_noised_gate_logits / torch.exp(self.softmax_temp)

        if self.use_sinkhorn:
            raw_gates = sinkhorn(maybe_noised_gate_logits, n_iters=self.sinkhorn_iters)
        else:
            raw_gates = maybe_noised_gate_logits.softmax(dim = -1)

        # find top N experts per position

        topk_return = self.topk(raw_gates, k = top_n)

        gate_indices = topk_return.indices  # b n k 

        if self.differentiable_topk:
            # allow for differentiable topk using coordinate descent
            # used successfully for routing from CoLT5 paper https://github.com/lucidrains/CoLT5-attention

            gates = topk_return.coor_descent_values
        else:
            gates = topk_return.values

        # move the top-n dimension to be first

        gates = rearrange(gates, '... k -> k ...')
        gate_indices = rearrange(gate_indices, '... k -> k ...')

        # masks

        one_hot_gate_indices = F.one_hot(gate_indices, self.num_experts) #(k,b,n,e)
        mask = one_hot_gate_indices.float()

        mask_1 = mask[0] # needed for balancing loss 每个token对应的最高专家得分的onehot编码

        # normalize top-n gate scores

        denom = reduce(gates, 'k ... -> 1 ...', 'sum').clamp(min = eps) #(1,b,n) 每个token对应专家的总分
        gates = gates / denom ##分数归一化(算是吧)

        # best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
        # optimal threshold was ~ 0.2
        # generalized to more than 2 experts

        probs = torch.zeros_like(gates).uniform_(0., 1.) #(k,b,n) 

        should_route = probs < einx.divide('k b n, k -> k b n', gates, threshold.clamp(min = eps)) #动态路由决定每个token对应的专家是否值得激活

        # tokens should always be routed to first expert
        # threshold for first expert already set to very small number, but just in case

        should_route[0, ...] = True #第一个专家一定要激活

        mask *= rearrange(should_route.float(), '... -> ... 1') #(k,b,n,1)和(k,b,n,e)相乘，来决定是否每个专家值得被激活，如果不激活，该专家的onehot编码就变成0向量

        mask_cumsum = cumsum_exclusive(mask, dim = -2) # along sequence dimension 统计每个 expert 当前被分配了多少 token（即 slot index），以便实现容量限制

        # compute assignment to experts - (batch, seq, experts)

        # This is the position within the expert's mini-batch for this sequence

        positions = []
        prev_expert_count = 0.

        for n in range(self.top_n):
            position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n] 

            # Remove the elements that don't fit. (batch, sequence, experts)
            mask[n] *= (position_in_expert < expert_capacity_f).float() 

            # How many examples in this sequence go to this expert - needed for the next iteration as offset
            prev_expert_count = reduce(mask[n], '... n e -> ... 1 e', 'sum') + prev_expert_count

            # (batch, sequence)
            position_in_expert = reduce(position_in_expert, '... n e -> ... n', 'sum') 
            positions.append(position_in_expert) #topk专家中，每个token对应专家处理的位置 

        positions = torch.stack(positions) #(k,b,n,e)

        # (k, batch, sequence) - mostly ones, but zeros where something didn't fit
        mask_flat = reduce(mask, '... n e -> ... n', 'sum') #统计每个token被选中了几次

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

        router_z_loss = self.zero
        balance_loss = self.zero

        if self.training:
            density_1 = reduce(mask_1, '... n e -> ... e', 'mean')
            density_1_proxy = reduce(raw_gates, '... n e -> ... e', 'mean') # Something continuous that is correlated with what we want to equalize.
            balance_loss = (density_1_proxy * density_1).mean() * float(self.num_experts ** 2)

            router_z_loss = torch.logsumexp(gate_logits, dim = -1)
            router_z_loss = torch.square(router_z_loss)            
            router_z_loss = router_z_loss.mean()

        return dispatch_tensor, combine_tensor, balance_loss, router_z_loss


class ExpertChoiceGating(nn.Module):
    """
    Each expert selects its top-k tokens. This ensures perfect load balancing by design.
    The number of tokens each expert processes (its capacity) is determined by `capacity_factor`.
    """

    def __init__(
        self,
        subset_experts_mapping: Dict[float, int] = {1.0: 1},
        add_noise: bool = False,
        noise_mult: float = 1.0,
        softmax_temp: List[float] = [0.5, 0.5],
        #
        capacity_factor_train=2.0,
        capacity_factor_eval=2.0,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True,
        # 
        use_sinkhorn = False,
        sinkhorn_iters: int = 6,
        **kwargs: Any,
    ):
        super().__init__()
        self.num_experts = sum(subset_experts_mapping.values())
        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.add_noise = add_noise
        self.noise_mult = noise_mult

        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iters  = sinkhorn_iters
        
        # Capacity factors determine how many tokens each expert will process
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

        self.softmax_temp = nn.Parameter(torch.log(torch.tensor(softmax_temp[-1], dtype=torch.float)))
        
        # A buffer for returning zero loss during evaluation
        self.register_buffer('zero', torch.zeros(1), persistent=False)

    def _get_expert_capacity(self, num_tokens):
        """Calculates the number of tokens each expert will process."""
        capacity_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        
        # Capacity is the number of tokens divided by experts, scaled by the capacity factor
        capacity = (num_tokens * capacity_factor) / self.num_experts
        capacity = max(capacity, MIN_EXPERT_CAPACITY)
        return int(capacity)

    def forward(self, key, query):
        """
        einstein notation:
        b - batch
        n - sequence length (tokens)
        e - experts
        c - capacity (tokens per expert)
        d - dimension
        """
        
        # Get tensor properties
        *_, n, d, dtype, device = *key.shape, key.dtype, key.device

        # 1. Calculate gate logits: each token gets a score for each expert
        # Input: (b, n, d) -> Output: (b, n, e)
        gate_logits = torch.einsum("b n d, e d -> b n e", key, query)

        maybe_noised_gate_logits = gate_logits

        if self.add_noise and self.training:
            noise_std = (1.0 / self.num_experts) * self.noise_mult
            noise = torch.randn_like(maybe_noised_gate_logits) * noise_std
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise

        # 2. Transpose for expert's perspective: experts now look across tokens
        # (b, n, e) -> (b, e, n)
        affinity_logits = rearrange(maybe_noised_gate_logits, 'b n e -> b e n')
        affinity_logits = affinity_logits / torch.exp(self.softmax_temp)

        # 3. Expert's Choice: Each expert selects its top-k tokens

        expert_capacity = self._get_expert_capacity(n)

        if self.use_sinkhorn:
            affinity_for_topk = sinkhorn(affinity_logits, n_iters=self.sinkhorn_iters)
        else:
            affinity_for_topk = affinity_logits.softmax(dim = -1)
        
        _, topk_indices = torch.topk(
            affinity_for_topk, k=expert_capacity, dim=-1
        )
        
        # 4. Create dispatch and combine tensors

        # (b, e, c, n) (b, n, e, c)
        dispatch_mask = F.one_hot(topk_indices, num_classes=n)
        dispatch_tensor = dispatch_mask.permute(0, 3, 1, 2).to(dtype)

        combine_tensor = torch.einsum('b n e c, b e n -> b n e c', dispatch_tensor, affinity_for_topk)

        if self.straight_through_dispatch_tensor:
            dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()

        # 5. Calculate Auxiliary Losses
        balance_loss = self.zero
        router_z_loss = self.zero

        # Router Z-Loss: Still useful.
        # This loss encourages the sum of logits for each token to be small,
        # which helps with training stability.
        # if self.training:
        #     # We compute it on the original (b, n, e) logits
        #     router_z_loss = torch.logsumexp(gate_logits, dim=-1)
        #     router_z_loss = torch.square(router_z_loss)
        #     router_z_loss = router_z_loss.mean()

            
        return dispatch_tensor, combine_tensor, balance_loss, router_z_loss


class SoftExpertChoiceGating(nn.Module):
    """
    Each expert selects its top-k tokens. This ensures perfect load balancing by design.
    The number of tokens each expert processes (its capacity) is determined by `capacity_factor`.
    """

    def __init__(
        self,
        subset_experts_mapping: Dict[float, int] = {1.0: 1},
        add_noise: bool = False,
        noise_mult: float = 1.0,
        softmax_temp: List[float] = [0.5, 0.5],
        #
        slots_per_expert: int = 1,
        selective: bool = True,
        dim: int = 192, 
        use_diverse_loss: bool = True,
        #
        use_sinkhorn = False,
        sinkhorn_iters: int = 6,
        **kwargs: Any,
    ):
        super().__init__()
        # Basic configs
        self.dim = dim
        self.add_noise = add_noise
        self.noise_mult = noise_mult
        self.selective = selective

        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iters  = sinkhorn_iters

        # Experts definition
        self.subset_experts_mapping = subset_experts_mapping
        self.num_experts = sum(self.subset_experts_mapping.values())
        self.slots_per_expert = slots_per_expert

        # Temperature parameters for logits
        self.softmax_temp = nn.ParameterList(
            [nn.Parameter(torch.log(torch.tensor(t, dtype=torch.float))) for t in softmax_temp]
        )

        # Selective masking modules
        if self.selective:
            mask_dim = 32
            self.mask_qproj = nn.Linear(self.dim, mask_dim)
            self.mask_kproj = nn.Sequential(nn.GELU(), nn.Linear(self.dim, mask_dim))
            lambda_dim = mask_dim
            self.lambda_q1 = nn.Parameter(torch.empty(lambda_dim).normal_(mean=0, std=0.1))
            self.lambda_k1 = nn.Parameter(torch.empty(lambda_dim).normal_(mean=0, std=0.1))
            self.lambda_q2 = nn.Parameter(torch.empty(lambda_dim).normal_(mean=0, std=0.1))
            self.lambda_k2 = nn.Parameter(torch.empty(lambda_dim).normal_(mean=0, std=0.1))
        
        # A buffer for returning zero loss during evaluation
        self.use_diverse_loss = use_diverse_loss
        self.register_buffer('zero', torch.zeros(1), persistent=False)

    def forward(self, key, query):
        """
        einstein notation:
        b - batch
        n - sequence length (tokens)
        e - experts
        c - capacity (tokens per expert)
        d - dimension
        """

        gate_logits = torch.einsum("b n d, e s d -> b n e s", key, query)
        maybe_noised_gate_logits = gate_logits

        if self.training and self.add_noise:
            noise_std = (1.0 / self.num_experts) * self.noise_mult
            noise = torch.randn_like(maybe_noised_gate_logits) * noise_std
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise

        # Temperatures for dispatch and combine
        dispatch_logits = maybe_noised_gate_logits / torch.exp(self.softmax_temp[0])
        combine_logits = maybe_noised_gate_logits / torch.exp(self.softmax_temp[1])

        if self.use_sinkhorn:
            dispatch_tensor = rearrange(dispatch_logits.flatten(start_dim=-2), 'b n es -> b es n')
            dispatch_tensor = sinkhorn(dispatch_tensor, n_iters=self.sinkhorn_iters)
            dispatch_tensor = rearrange(dispatch_tensor, 'b (e s) n -> b n e s', e=self.num_experts, s=self.slots_per_expert)
        else:
            dispatch_tensor = torch.softmax(dispatch_logits, dim=-3)

        combine_tensor = torch.softmax(combine_logits.flatten(start_dim=-2), dim=-1)
        combine_tensor = rearrange(combine_tensor, 'b n (e s) -> b n e s', e=self.num_experts, s=self.slots_per_expert)

        if self.selective:
            noise_logits = torch.einsum(
                "b n d, e s d -> b n e s",
                self.mask_kproj(key),
                self.mask_qproj(query),
            )
            noise_logits = noise_logits / torch.exp(self.softmax_temp[0])
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
            lambda_full = lambda_1 - lambda_2
            noise_tensor = torch.softmax(noise_logits, dim=-3)
            dispatch_tensor = dispatch_tensor - lambda_full * noise_tensor

            # renorm
            dispatch_tensor = dispatch_tensor.clamp_min(1e-6)
            row_sum = dispatch_tensor.sum(dim=-3, keepdim=True)
            dispatch_tensor = dispatch_tensor / row_sum

        # 5. Calculate Auxiliary Losses
        diverse_loss = self.zero
        router_z_loss = self.zero

        # if self.training and self.use_diverse_loss and self.slots_per_expert > 1:
        #     # 步骤 1: 计算相似度矩阵 (假设query已归一化，如果没有需要先归一化)
        #     normalized_queries = F.normalize(query, p=2, dim=-1) # 确保是余弦相似度
        #     similarity_matrix = torch.einsum('esd, etd -> est', normalized_queries, normalized_queries)
            
        #     # 步骤 2: 提取上三角（不含对角线）的相似度值
        #     s = self.slots_per_expert
        #     triu_mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=query.device), diagonal=1)
        #     off_diagonal_similarities = similarity_matrix[:, triu_mask]
            
        #     # 步骤 3: 应用带边距的Hinge Loss
        #     hinge_loss = F.relu(off_diagonal_similarities - 0.0).mean()
            
        #     # 步骤 4: 计算最终的平均损失
        #     diverse_loss = hinge_loss
            
        return dispatch_tensor, combine_tensor, diverse_loss, router_z_loss


class MoE(nn.Module):
    def __init__(        
        self,
        dim: int,
        subset_experts_mapping: Dict[float, int] = {1.0: 1},
        add_noise: bool = False,
        noise_mult: float = 1.0,
        moe_droprate: float = 0.0,
        moe_droprate_act: Optional[float] = None,
        softmax_temp: List[float] = [0.5, 0.5],
        block_depth: int = 0,
        compute_similarity_metrics = False,
        balance_loss_coef = 1e-2,
        router_z_loss_coef = 1e-3,
        global_expert = True,
        router_type = 'Soft',  # 'TC' 'EC' 'Soft'
        router_qk_norm: bool = True,
        query_layernorm: bool = False,
        use_botneck: bool = False,
        w1_as_query: bool = True,
        #
        slots_per_expert: int = 1,
        selective: bool = True,
        #
        threshold_train = 0.2,
        threshold_eval = 0.2,
        gating_top_n = 2,
        #
        capacity_factor_train = 2.0,
        capacity_factor_eval = 2.0,
        straight_through_dispatch_tensor = True,
        differentiable_topk = False,
        differentiable_topk_fused = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.dim = dim
        self.subset_experts_mapping = subset_experts_mapping
        self.num_experts = sum(self.subset_experts_mapping.values())
        self.slots_per_expert = slots_per_expert if router_type == 'Soft' else 1
        
        # Gating options
        self.query_layernorm_flag = query_layernorm

        self.key_downsample = nn.Identity()
        if use_botneck:
            qk_dim = 64
            self.key_downsample = nn.Linear(dim, qk_dim, bias=False)
        else:
            qk_dim = dim

        self.key_norm = nn.Identity()
        self.query_norm = nn.Identity()
        if router_qk_norm:
            self.key_norm = L2Norm(dim=-1)
            self.query_norm = (nn.Sequential(nn.LayerNorm(qk_dim), L2Norm(dim=-1)) if self.query_layernorm_flag else L2Norm(dim=-1))

        self.w1_as_query = w1_as_query
        if w1_as_query:
            self.w1_to_query = nn.ModuleDict()
            for ratio, num_experts in self.subset_experts_mapping.items():
                expert_key = f"{get_ratio_str(ratio)}"
                hidden_dim = int(4 * ratio * dim)
                self.w1_to_query[expert_key] = MeanExpander(self.slots_per_expert, learnable=True, dim=hidden_dim)
        else:
            query_init = self.initialize_query(self.num_experts, self.slots_per_expert, qk_dim)
            self.query = nn.Parameter(query_init)

        self.slot_norm = nn.Identity()
        if router_type == 'EC':
            self.router_func = ExpertChoiceGating
        elif router_type == 'TC':
            self.router_func = TokenChoiceGating
        else:
            self.router_func = SoftExpertChoiceGating
            # self.slot_norm = nn.LayerNorm(dim)

        self.router = self.router_func(
            subset_experts_mapping = subset_experts_mapping,
            add_noise = add_noise,
            noise_mult = noise_mult,
            softmax_temp  = softmax_temp,
            #
            slots_per_expert = self.slots_per_expert,
            selective = selective,
            dim = qk_dim,
            #
            top_n = gating_top_n,
            threshold_train = threshold_train,
            threshold_eval = threshold_eval,
            #
            capacity_factor_train = capacity_factor_train,
            capacity_factor_eval = capacity_factor_eval,
            straight_through_dispatch_tensor = straight_through_dispatch_tensor,
            differentiable_topk = differentiable_topk,
            differentiable_topk_fused = differentiable_topk_fused,
        )

        self.experts = nn.ModuleDict()
        for ratio, num_experts in self.subset_experts_mapping.items():
            expert_key = f"{get_ratio_str(ratio)}"
            hidden_dim = int(4 * ratio * dim)
            self.experts[expert_key] = MultiExpertLayer(
                in_dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                moe_droprate=moe_droprate[ratio],
                moe_droprate_act=moe_droprate_act,
            )

        self.global_expert = global_expert
        if self.global_expert:
            # self.pad = FixedLearnablePad(input_len=int(self.num_experts * self.slots_per_expert), dim=dim)
            self.pad = DynamicPad()
            self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
            self.slots_proj = EnhancedKroneckerLinear(dim, dim)
            self.dwc_norm = nn.LayerNorm(dim)
            self.gate = nn.SiLU()

        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.compute_similarity_metrics = compute_similarity_metrics
    
    def initialize_query(self, num_experts: int, slots_per_expert: int, d_model: int, init_method: str = ''):
        """Orthogonal initialization for expert queries."""
        query_init = torch.empty(num_experts, slots_per_expert, d_model)
        if init_method == 'orthogonal':
            nn.init.orthogonal_(query_init)
        elif init_method == 'kaiming':
            nn.init.kaiming_uniform_(query_init, a=math.sqrt(5))
        else:
            nn.init.trunc_normal_(query_init, std=0.02)
        return query_init

    def forward(self, x):

        query = None
        if self.w1_as_query:
            query_list = [
                w2q(self.experts[expert_key].fc1.weight.transpose(-1, -2)).transpose(-1, -2)
                for expert_key, w2q in self.w1_to_query.items()
            ]
            query = torch.stack(query_list, dim=0).view(-1, self.slots_per_expert, self.dim)
        else:
            query = self.query

        if not isinstance(self.router, SoftExpertChoiceGating):
            query = query.squeeze(-2)

        query = self.query_norm(query)
        key = self.key_norm(self.key_downsample(x))
        
        dispatch_tensor, combine_tensor, balance_loss, router_z_loss = self.router(key, query)

        # dispatch
        slots = torch.einsum("... n d, ... n e s -> ... e s d", x, dispatch_tensor)
        slots = self.slot_norm(slots)

        # Residual path
        if self.global_expert:
            res_slots = rearrange(slots, "... e s d -> ... (e s) d")
            res_slots, h, w = self.pad(res_slots)
            res_slots = self.dwc(rearrange(res_slots, "... (h w) d -> ... d h w", h=h, w=w))
            res_slots = rearrange(res_slots, "... d h w -> ... (h w) d")
            res_slots = self.slots_proj(self.gate(self.dwc_norm(res_slots)))
            res_slots = self.pad.unpad(res_slots)
            res_slots = rearrange(res_slots, "... (e s) d -> ... e s d", e=self.num_experts)

        # Expert transformations
        expert_slots = torch.split(slots, [expert.num_experts for expert in self.experts.values()], dim=-3)
        outputs = [expert_module(subset_slots) for subset_slots, expert_module in zip(expert_slots, self.experts.values())]
        outputs = torch.cat(outputs, dim=1)

        # combine
        if self.global_expert:
            outputs = res_slots + outputs
        output = torch.einsum('... e s d, ... n e s -> ... n d', outputs, combine_tensor)

        # losses

        weighted_balance_loss = balance_loss * self.balance_loss_coef
        weighted_router_z_loss = router_z_loss * self.router_z_loss_coef

        # combine the losses

        total_aux_loss = weighted_balance_loss + weighted_router_z_loss

        metrics = None
        if self.compute_similarity_metrics:
            metrics = {}
            # combine weights 相似度（token 级）
            cw_reshaped = combine_tensor.view(b, n, self.num_experts, s)        # (b, n, e, s)
            cw_sim = cosine_psim(cw_reshaped, batch_dims=(0,), contract_dims=(2, 3))
            eye_tok = torch.eye(n, device=x.device).unsqueeze(0)
            cw_mask = cw_sim * (1 - eye_tok) + eye_tok * cw_sim.amax()
            metrics["combine_tensor_similarity_min"] = cw_mask.amin()
            metrics["combine_tensor_similarity_max"] = cw_mask.amax()
            metrics["combine_tensor_similarity_mean"] = (cw_sim.sum() - b * n) / (b * n * (n - 1))

            # dispatch weights 相似度（slot 级）
            dw_sim = cosine_psim(dispatch_tensor, batch_dims=(0,), contract_dims=(1,))
            eye_slot = torch.eye(self.num_experts * s, device=x.device).view(
                1, self.num_experts, s, self.num_experts, s
            )
            dw_mask = dw_sim * (1 - eye_slot) + eye_slot * dw_sim.amax()
            metrics["dispatch_tensor_similarity_min"] = dw_mask.amin()
            metrics["dispatch_tensor_similarity_max"] = dw_mask.amax()
            metrics["dispatch_tensor_similarity_mean"] = (
                (dw_sim.sum() - b * self.num_experts * s)
                / (b * self.num_experts * s * (self.num_experts * s - 1))
            )

            # prototype 相似度（query 本身）
            prototypes = query                                                    # (e, s, d)
            mu_sim = cosine_psim(prototypes, batch_dims=(), contract_dims=(2,))
            eye_mu = torch.eye(self.num_experts * s, device=x.device).view(self.num_experts, s, self.num_experts, s)
            mu_mask = mu_sim * (1 - eye_mu) + eye_mu * mu_sim.amax()
            metrics["mu_similarity_min"] = mu_mask.amin()
            metrics["mu_similarity_max"] = mu_mask.amax()
            metrics["mu_similarity_mean"] = (
                (mu_sim.sum() - self.num_experts * s) / (self.num_experts * s * (self.num_experts * s - 1))
            )
            
        return output, total_aux_loss


class MultiExpertLayer(nn.Module):
    """A more efficient alternative to creating 'n' separate expert layers (likely
    from 'nn.Linear' modules).  Instead, we create a single set of batched weights
    and biases, and apply all 'experts' in parallel.

    Args:
        embed_dim (int): embedding dimension (d)
        num_experts (int): number of experts (n)
        bias (bool): whether to include a bias term. Default: True
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_experts: int,
        moe_droprate: float = 0.0,
        moe_droprate_act: Optional[float] = None,
        act_fn: nn.Module = nn.GELU,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        norm_layer=None,
        layer_scale=False,
        freeze_moe=False,
    ):
        super().__init__()
        self.in_features = in_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.norm = norm_layer(hidden_dim) if norm_layer is not None else nn.Identity()
        
        self.fc1 = MultiExpertLinear(num_experts, in_dim, self.hidden_dim)
        self.act_fn = act_fn()
        self.fc2 = MultiExpertLinear(num_experts, self.hidden_dim, in_dim)

        
        self.layer_scale = layer_scale
        self.scale_in = OffsetScale(in_dim) if self.layer_scale else nn.Identity()
        self.scale_out = OffsetScale(in_dim) if self.layer_scale else nn.Identity()

        self.drop_1 = nn.Dropout(moe_droprate_act) if moe_droprate_act is not None else nn.Dropout(moe_droprate)
        self.drop_2 = nn.Dropout(moe_droprate)
        
        self.freeze_moe = freeze_moe
        if self.freeze_moe:
            for param in self.fc1.parameters():
                param.requires_grad = False
            for param in self.fc2.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.scale_in(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.drop_1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop_2(x)
        x = self.scale_out(x)
        return x

class MultiExpertLinear(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim, use_bias=True):
        super(MultiExpertLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_dim, input_dim))
        for i in range(self.weight.shape[0]):
            trunc_normal_(self.weight[i], std=.02)

        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)
        
    def forward(self, x):
        if x.dim() == 4:
            x = einsum(x, self.weight, "b e s d1, e d2 d1 -> b e s d2")
            x = x + rearrange(self.bias, "e d2 -> () e () d2") if self.use_bias else x
        else:
            x = einsum(x, self.weight, "b e d1, e d2 d1 -> b e d2")
            x = x + rearrange(self.bias, "e d2 -> () e d2") if self.use_bias else x
        return x
