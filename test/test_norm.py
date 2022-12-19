import math
import torch
# import numpy as np
# import sys, os
from torch import nn, Tensor
# from torch.nn import functional as F
from typing_extensions import Final
from typing import List
import math


def get_norm_alpha(sr: int = 48000,
                   hop_size: int = 480,
                   tau: float = 1,
                   log: bool = True):

    a_ = _calculate_norm_alpha(sr=sr, hop_size=hop_size, tau=tau)
    precision = 3
    a = 1.0
    while a >= 1.0:
        a = round(a_, precision)
        precision += 1
    if log:
        print(f"Running with normalization window alpha = '{a}'")
    return a


def _calculate_norm_alpha(sr: int, hop_size: int, tau: float):
    dt = hop_size / sr
    return math.exp(-dt / tau)


class ExponentialUnitNorm(nn.Module):

    alpha: Final[float]
    eps: Final[float]

    def __init__(self, alpha: float, num_freq_bins: int, eps: float = 1e-14):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.init_state: Tensor
        self.UNIT_NORN_INIT = [0.001, 0.0001]
        self.unit_norm_init = torch.linspace(self.UNIT_NORN_INIT[0],
                                             self.UNIT_NORN_INIT[1],
                                             num_freq_bins).unsqueeze(0)
        s = self.unit_norm_init
        s = s.view(1, 1, num_freq_bins, 1)
        self.register_buffer("init_state", s)
        # s = torch.from_numpy()

    def forward(self, x: Tensor):
        b, c, t, f, _ = x.shape
        x_abs = x.square().sum(dim=-1, keepdim=True).clamp_min(
            self.eps).sqrt()  # square sum
        state = self.init_state.clone().expand(b, c, f, 1)
        out_state: List[Tensor] = []
        for t in range(t):
            state = x_abs[:, :, t] * (1 - self.alpha) + state * self.alpha
            out_state.append(state)
        return x / torch.stack(out_state, 2).sqrt()


if __name__ == "__main__":
    """
    test norm method
    """
    F1 = 96
    sr = 48000
    hop_size = 480
    tau = 1.
    alpha = get_norm_alpha(log=False, sr=sr, hop_size=hop_size, tau=tau)
    tmp = ExponentialUnitNorm(0.8, 96)
    spec = torch.randn(2, 1, 100, F1, 2)
    x = tmp(spec)
    norm_torch = torch.view_as_complex(x.squeeze(1))
    print(f"norm_torch :{norm_torch}")
    print('sc')
