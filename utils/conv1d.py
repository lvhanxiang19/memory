import torch
from torch import nn
from torch.nn import functional as F


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation  # 保证因果卷积，不越界未来信息
        self.in_channel=in_channels
        self.out_channel=out_channels
        # depthwise卷积：groups=in_channels
        self.depthwise =nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        if in_channels!=out_channels:
        # pointwise卷积：1x1卷积，将in_channels转换为out_channels
         self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        # 输入x：形状 (b, s, d1)
        # 1d卷积的输入通常形状是 (b, c, seq_len)，所以先需要转置
        x = x.transpose(1,2)  # (b, d1, s)

        # depthwise conv
        x = self.depthwise(x)

        # 去除多余的padding，防止未来信息泄露
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        if self.in_channel!=self.out_channel:
        # pointwise conv
         x = self.pointwise(x)

        # 再转置回来形状 (b, s, d2)
        x = x.transpose(1,2)
        return x