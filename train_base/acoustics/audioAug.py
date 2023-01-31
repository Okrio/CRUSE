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

def compositeSecFilt(indata,filter_num=3, sr=16000):
    filter_list = [
        "high_shelf", "high_pass", "low_shelf", "low_pass", "peaking_eq",
        "notch"
    ]
    assert filter_num < len(filter_list), " filter_num is error"
    filter_idx = list(np.linspace(0, 5, 6, dtype=np.int16))
    sel_filter = random.sample(filter_idx, filter_num)
    indata_tmp = indata
    for i in range(0,filter_num):
        filter_type = filter_list[sel_filter[i]]
        center_freq = ss.loguniform.rvs(REGISTERED_SecFilter_freq[filter_type][0], REGISTERED_SecFilter_freq[filter_type][1], 1)
        gain_db = np.random.uniform(-15,15, 1)
        q_factor = np.random.uniform(0.5, 1.5, 1)
        selFilt_coef = REGISTERED_SecFilter[filter_type](center_freq,gain_db,q_factor,sr)
        indata_tmp = torchaudio.functional.lfilter(indata_tmp, selFilt_coef[1,:], selFilt_coef[0,:])
    return indata_tmp


def hp_filter(indata,filte_num=1, sr=16000):
    """
    fixed frequency high pass filter
    """
    center_freq = 150.
    q_factor = np.random.uniform(0.5,1.5,1)
    filt_coef = high_pass(center_freq,0,q_factor,sr)
    out = indata
    for i in range(0, filte_num):
        out = torchaudio.functional.lfilter(out, filt_coef[1,:],filt_coef[0,:])
    return out

def airAbsorption(sig, sr=16000):
    center_freq = [125,250,500,1000,2000,4000,8000, 16000,24000]
    air_absorption = [0.1,0.2,0.5,1.1,2.7,9.4,29.0,91.5,289.0]
    air_absorption_table = Tensor([x * 1e-3 for x in air_absorption])
    distance_low = 1.0
    distance_high = 20.0
    d = torch.FloatTensor(1).uniform_(distance_low,distance_high)
    atten_val = torch.exp(-d * air_absorption_table)
    atten_val_db = 20 * torch.log10(atten_val)
    att_interp_db = interp_atten(att_interp_db, 161)
    att_interp = 10 ** (att_interp_db / 20)
    sig_stft = torch.stft(sig, window=torch.hann_window(320), n_fft=320, win_length=320, hop_length=160, return_complex=True).squeeze()
    att_interp_tile = torch.tile(att_interp, (sig_stft.shape[-1], 1)).transpose(1,0)
    masked = sig_stft * att_interp_tile
    masked = masked.unsqueeze()
    rc = torch.istft(masked, window = torch.hann_window(320), n_fft=320, win_length=320,hop_length=320, length=sig.shape[-1])
    return rc

def interp_atten(atten_vals=None, n_freq=None, center_freq=None, sr=16000):
    center_freq = [125,250,500, 1000, 2000, 4000, 8000, 16000, 24000]
    sr=16000
    atten_vals1 = [atten_vals[0].tolist()] + atten_vals.tolist() + [atten_vals[-1].tolist()]
    freqs = torch.linspace(0, sr/2, n_freq)
    atten_vals_interp = torch.zeros(n_freq)
    center_freq = [0] + center_freq + [sr/2]
    i = 0
    center_freq_win = as_windowed(Tensor([center_freq]), 2,1).squeeze()
    atten_vals_win = as_windowed(Tensor([atten_vals1]), 2,1).squeeze()
    gf = center_freq_win.tolist()
    for k,(c,a) in enumerate(zip(center_freq_win.tolist(), atten_vals_win.tolist())):
        c0, c1 = c[0], c[1]
        a0, a1 = a[0], a[1]
        while i <n_freq and freqs[i] <=c1:
            x = (freqs[i] - c1) / (c0 -c1)
            atten_vals_interp[i] = a0 *x + a1 * (1.-x)
            i+=1
    return atten_vals_interp

def as_windowed(x:torch.Tensor, win_len, hop_len=1, dim=1):
    shape:List[int] = list(x.shape)
    stride:List[int] = list(x.stride())
    shape[dim] = int((shape[1] - win_len + hop_len) // hop_len)
    shape.insert(dim+1, win_len)
    stride.insert(dim+1, stride[dim])
    stride[dim] = stride[dim] * hop_len
    y=x.as_strided(shape, stride)
    return y

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
