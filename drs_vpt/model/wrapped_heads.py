# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

import torch
import torch.nn as nn




class ResidualPointwise(nn.Module):
    """Identity-at-init per-location MLP: y = x + W2(act(W1(x))) with W2=0 at init."""
    def __init__(self, ch: int, hidden: int | None = None, act: str = "gelu"):
        super().__init__()
        hidden = hidden or ch
        self.fc1 = nn.Conv2d(ch, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, ch, 1, bias=True)

        if act == "gelu":
            self.act = nn.GELU()
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown act={act}")

        # residual branch starts at 0 => block is identity at init
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class WrappedConv1x1(nn.Module):
    """
    Generic wrapper around a Conv2d(in_ch->out_ch, 1x1) that adds expressive
    pointwise residual MLP blocks before/after while preserving the base mapping
    exactly at init.

    x -> pre(identity) -> base(conv1x1) -> post(identity)
    """
    def __init__(
        self,
        base: nn.Conv2d,
        *,
        pre_blocks: int = 2,
        post_blocks: int = 2,
        pre_hidden: int | None = None,
        post_hidden: int | None = None,
        act: str = "gelu",
    ):
        super().__init__()
        assert isinstance(base, nn.Conv2d)
        assert base.kernel_size == (1, 1)

        in_ch = base.in_channels
        out_ch = base.out_channels
        pre_hidden = pre_hidden or in_ch
        post_hidden = post_hidden or max(out_ch, 64)  # sensible default

        self.pre = nn.Sequential(*[
            ResidualPointwise(in_ch, hidden=pre_hidden, act=act)
            for _ in range(pre_blocks)
        ])
        self.base = base
        self.post = nn.Sequential(*[
            ResidualPointwise(out_ch, hidden=post_hidden, act=act)
            for _ in range(post_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post(self.base(self.pre(x)))


def make_wrapped_conv1x1(
    in_channels: int,
    out_channels: int,
    *,
    pre_blocks: int = 2,
    post_blocks: int = 2,
    pre_hidden: int | None = None,
    post_hidden: int | None = None,
    act: str = "gelu",
    bias: bool = True,
) -> WrappedConv1x1:
    base = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
    return WrappedConv1x1(
        base,
        pre_blocks=pre_blocks,
        post_blocks=post_blocks,
        pre_hidden=pre_hidden,
        post_hidden=post_hidden,
        act=act,
    )
