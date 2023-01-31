from itertools import filterfalse
import torch
import torchaudio
# import os
import numpy as np
import scipy.stats as ss
# from scipy.stat import loguniform
from typing import List
from torch import Tensor
import random


def high_shelf(center_freq: Tensor, gain_db: Tensor, q_factor: Tensor,
               sr: float):
    w0 = Tensor([2. * np.pi * center_freq / sr])
    amp = torch.pow(10, gain_db / 40.)
    alpha = torch.sin(w0) / 2. / q_factor
    b0 = amp * ((amp + 1) +
                (amp - 1) * torch.cos(w0) + 2 * torch.sqrt(amp) * alpha)
    b1 = -2 * amp * ((amp - 1) + (amp + 1) * torch.cos(w0))
    b2 = amp * ((amp + 1) +
                (amp - 1) * torch.cos(w0) - 2 * torch.sqrt(amp) * alpha)
    a0 = (amp + 1) - (amp - 1) * torch.cos(w0) + 2 * torch.sqrt(amp) * alpha
    a1 = 2 * ((amp - 1) - (amp + 1) * torch.cos(w0))
    a2 = (amp + 1) - (amp - 1) * torch.cos(w0) - 2 * torch.sqrt(amp) * alpha

    b = torch.cat((b0, b1, b2), dim=-1)
    a = torch.cat((a0, a1, a2), -1)
    coef = torch.cat((b, a), 0)
    return coef


def high_pass(center_freq: Tensor, gain_db: Tensor, q_factor: Tensor, sr: Tensor):

    w0 = Tensor([2. * np.pi * center_freq / sr])
    alpha = torch.sin(w0) / 2. / q_factor

    b0 = (1 + torch.cos(w0)) / 2.
    b1 = -(1 + torch.cos(w0))
    b2 = b0

    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha

    b = torch.cat((b0, b1, b2))
    a = torch.cat((a0, a1, a2))
    coef = torch.stack((b, a), 0)
    return coef


def low_shelf(center_freq: Tensor, gain_db: Tensor, q_factor: Tensor,
              sr: float):
    w0 = Tensor([2. * np.pi * center_freq / sr])
    amp = torch.pow(10, gain_db / 40.)
    alpha = torch.sin(w0) / 2. / q_factor

    b0 = amp * ((amp + 1) -
                (amp - 1) * torch.cos(w0) + 2 * torch.sqrt(amp) * alpha)
    b1 = 2 * amp * ((amp - 1) - (amp + 1) * torch.cos(w0))
    b2 = amp * ((amp + 1) -
                (amp - 1) * torch.cos(w0) - 2 * torch.sqrt(amp) * alpha)
    a0 = (amp + 1) + (amp - 1) * torch.cos(w0) + 2 * torch.sqrt(amp) * alpha
    a1 = -2 * ((amp - 1) + (amp + 1) * torch.cos(w0))
    a2 = (amp + 1) + (amp - 1) * torch.cos(w0) - 2 * torch.sqrt(amp) * alpha

    b = torch.cat((b0, b1, b2), -1)
    a = torch.cat((a0, a1, a2), -1)
    coef = torch.cat((b, a), 0)
    return coef


def low_pass(center_freq: Tensor, gain_db: Tensor,  q_factor: Tensor, sr: float):
    w0 = Tensor([2. * np.pi * center_freq / sr])
    alpha = torch.sin(w0) / 2. / q_factor

    b0 = (1 - torch.cos(w0)) / 2
    b1 = 1 - torch.cos(w0)
    b2 = b0

    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha

    b = torch.cat((b0, b1, b2))
    a = torch.cat((a0, a1, a2))

    coef = torch.stack((b, a), 0)
    return coef


def peaking_eq(center_freq: Tensor, gain_db: Tensor, q_factor: Tensor,
               sr: float):

    w0 = Tensor([2. * np.pi * center_freq / sr])
    amp = torch.pow(10, gain_db / 40.)
    alpha = torch.sin(w0) / 2. / q_factor

    b0 = 1 + alpha * amp
    b1 = -2 * torch.cos(w0)
    b2 = 1 - alpha * amp

    a0 = 1 + alpha / amp
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha / amp

    b = torch.cat((b0, b1, b2))
    a = torch.cat((a0, a1, a2))
    coef = torch.stack((b, a), 0)
    return coef


def notch(center_freq: Tensor, gain_db: Tensor, q_factor: Tensor, sr: float):
    w0 = Tensor([2. * np.pi * center_freq / sr])
    alpha = torch.sin(w0) / 2. / q_factor

    b0 = Tensor(1.)
    b1 = -2. * torch.cos(w0)
    b2 = b0

    a0 = 1. + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1. - alpha

    b = torch.cat((b0, b1, b2))
    a = torch.cat((a0, a1, a2))

    coef = torch.stack((b, a), 0)
    return coef


REGISTERED_SecFilter = {
    "high_shelf": high_shelf,
    "high_pass": high_pass,
    "low_shelf": low_shelf,
    "low_pass": low_pass,
    "peaking_eq": peaking_eq,
    "notch": notch
}
REGISTERED_SecFilter_freq = {
    "high_shelf": [1000,4000],
    "high_pass":[40, 400],
    "low_shelf":[40, 1000],
    "low_pass":[3000, 8000],
    "peaking_eq": [40, 4000],
    "notch": [40, 4000]
}

if __name__ == "__main__":
    filter_num = 3
    filter_list = [
        "high_shelf", "high_pass", "low_shelf", "low_pass", "peaking_eq",
        "notch"
    ]
    assert filter_num < len(filter_list), " filter_num is error"
    tmp = list(np.linspace(0, 5, 6, dtype=np.int16))
    sel_filter = random.sample(tmp, filter_num)
    sr = 16000
    for i in range(0, filter_num):
        filter_type = filter_list[i]
        center_freq = ss.loguniform.rvs(REGISTERED_SecFilter_freq[filter_type][0],REGISTERED_SecFilter_freq[filter_type][1], 1)
        gain_db = np.random.uniform(-15, 15, 1)
        q_factor = np.random.uniform(0.5,1.5, 1)
        out = REGISTERED_SecFilter[filter_type](center_freq, gain_db, q_factor,sr)




    print('sc')
