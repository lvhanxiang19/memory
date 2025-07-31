# Copyright (c) Meta Platforms, Inc. and affiliates
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributed._tensor import (
    DTensor,
)





import logging

logger = logging.getLogger()


class xFormerEmbeddingBag(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size, dim, dtype=torch.bfloat16))

    def forward(self, indices, scores):
        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
            num_shards = self.weight.device_mesh.size()
            if num_shards > 1:
                # scale gradients so that we end up with the average rather than sum
                grad_scale = 1 / num_shards
                weight = weight * grad_scale + (weight * (1-grad_scale)).detach()
        else:

            weight = self.weight
            with torch.cuda.amp.autocast(enabled=False):
              output = F.embedding_bag(indices, weight.to(torch.float32), per_sample_weights=scores.to(torch.float32), mode="sum")
            
        return output.to(torch.bfloat16)


