import torch
import torchaudio
# import os
import numpy as np
from scipy.stat import loguniform
from typing import List
from torch import Tensor


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


def high_pass(center_freq: Tensor, q_factor: Tensor, sr: Tensor):

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


def low_pass(center_freq: Tensor, q_factor: Tensor, sr: float):
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


def notch(center_freq: Tensor, q_factor: Tensor, sr: float):
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


def randFilt(thr: float = 0.375):
    r = torch.FloatTensor(4).uniform_(-thr, thr)

    b = torch.ones([1, 3])
    a = torch.ones_like(b)
    b[0, 1:] = r[:2]
    a[0, 1:] = r[2:]
    coef = torch.cat((b, a), 0)
    return coef


def randClipping(db_range=None, c_range=(0.01, 0.25), eps=1e-10, eps_c=0.001):
    pass


def suppress_late(rir: np.ndarray, sr: float, rt60: float, offset: int):
    len = rir.shape[-1]
    decay = torch.ones(1, len)
    dt = 1. / sr
    rt60_level = np.power(10., -60 / 20)
    tau = -rt60 / np.log10(rt60_level)
    if offset >= len:
        return rir
    for v in range(0, len - offset):
        decay[v] = np.exp(-v * dt / tau)

    rir = rir * decay
    return rir


def trim(rir: np.ndarray, ref_idx: int):
    min_db = -80
    len = rir.shape[-1]
    rir_mono = rir
    ref_level = rir_mono[ref_idx]
    min_level = np.power(10, (min_db + np.log10(ref_level) * 20.) / 20.)
    idx = len
    pass


def as_windowed(x: torch.Tensor, win_len, hop_len=1, dim=1):
    """
    input: B, T
    output: B, T//win_len, win_len
    """
    shape: List[int] = list(x.shape)
    stride: List[int] = list(x.stride())
    shape[dim] = int((shape[dim] - win_len + hop_len) // hop_len)
    shape.insert(dim + 1, win_len)
    stride.insert(dim + 1, stride[dim])
    stride[dim] = stride[dim] * hop_len
    y = x.as_strided(shape, stride)
    return y


def airAbsorption(sig, sr=16000):
    center_freqs = [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 24000]
    air_absorption = [0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0, 91.5, 289.0]
    air_absorption_table = [x * 1e-3 for x in air_absorption]
    distance_low = 1.0
    distance_high = 20.0
    d = np.random.uniform(distance_low, distance_high, 1)

    atten_vals = np.exp(-d * air_absorption_table)
    atten_vals_db = 20 * np.log10(atten_vals)
    atten_interp_db = interp_atten(atten_interp_db, 161)
    atten_interp = 10**(atten_interp_db / 20.)
    sig_stft = torch.stft(sig,
                          window=torch.hann_window(320),
                          n_fft=320,
                          win_length=320,
                          hop_length=160,
                          return_complex=True).squeeze()
    att_interp_tile = torch.tiel(atten_interp,
                                 (sig_stft.shape[-1], 1)).tranpose(1, 0)
    masked = sig_stft * att_interp_tile
    masked = masked.unsqeeze(0)
    rc = torch.istft(masked,
                     window=torch.hann_window(512),
                     n_fft=320,
                     win_length=320,
                     hop_length=160,
                     length=sig.shape[-1])
    torchaudio.save('ost_air.wav', rc, sr)


def interp_atten(atten_vals, n_freqs, center_freqs, sr=16000):
    atten_vals1 = atten_vals[0] + atten_vals + atten_vals[-1]
    freqs = np.linspace(0., sr / 2., n_freqs)
    atten_vals_interp = np.zeros(n_freqs)

    center_freqs = [0] + center_freqs + [sr / 2]
    i = 0
    center_freqs_win = as_windowed(Tensor([center_freqs]), 2, 1).squeeze()
    atten_vals_win = as_windowed(Tensor([atten_vals1]), 2, 1).squeeze()

    for k, (c, a) in enumerate(
            zip(center_freqs_win.tolist(), atten_vals_win.tolist())):
        c0, c1 = c[0], c[1]
        a0, a1 = a[0], a[1]
        while i < n_freqs and freqs[i] <= c1:
            x = (freqs[i] - c1) / (c0 - c1)
            atten_vals_interp[i] = a0 * x + a1 * (1. - x)
            i += 1
    return atten_vals_interp


if __name__ == "__main__":
    psthq = "ssf.wav"
    sig, sr = torchaudio.load(psthq)

    sr = 16000
    gain_db = torch.FloatTensor(1).uniform_(-15, 15)
    q_factor = torch.FloatTensor(1).uniform_(0.5, 1.5)
    # high_shelf
    center_freq1 = loguniform.rvs(1000, 6000, size=1)

    hs_coef = high_shelf(center_freq1, gain_db, q_factor, sr)
    out_hs = torchaudio.functional.lfilter(sig, hs_coef[1, :], hs_coef[0, :])
    torchaudio.save('ost.wav', out_hs, sr)
    # hig_pass
    center_freq1 = loguniform.rvs(40, 400, size=1)
    # low_shelf
    center_freq1 = loguniform.rvs(40, 1000, size=1)
    # low_pass
    center_freq1 = loguniform.rvs(3000, 8000, size=1)
    # peaking_eq
    center_freq1 = loguniform.rvs(40, 4000, size=1)
    # notch
    center_freq1 = loguniform.rvs(40, 4000, size=1)