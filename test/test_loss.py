'''
Author: your name
Date: 2022-03-02 22:19:22
LastEditTime: 2022-03-03 22:58:14
LastEditors: Please set LastEditors
Description: refer to "weighted speech distortion lossed for neural network based real-time speech enhancement"
FilePath: /CRUSE/test/test_loss.py
'''
import os

from matplotlib import pyplot as plt
import numpy as np
import librosa as lib
import soundfile as sf
import torch
import torch.nn.functional as F
from typing import Dict, Final, Iterable, List
from torch import Tensor, nn
from torch.autograd import Function


def plot_mesh(img, title="", save_home=""):
    img = img
    fig, ax = plt.subplots()
    plt.title(title)
    fig.colorbar(plt.pcolormesh(range(img.shape[1]), range(img.shape[0]), img))
    if save_home != "":
        print(os.path.join(save_home, "%s.jpg" % title))
        plt.savefig(os.path.join(save_home, "%s.jpg" % title))
        return


def snr_weight_loss_adjust():
    global_snr = np.linspace(-10, 30, 410)
    beta = np.array([1, 5, 10, 20])
    # global_snr = np.expand_dims(global_snr, axis=0)
    # beta = np.expand_dims(beta, axis=-1)
    alpha = np.zeros((len(beta), len(global_snr)))
    for i in range(len(beta)):
        tmp = 10**(global_snr / 10)
        beta_tmp = 10**(beta[i] / 10)
        alpha[i] = tmp / (tmp + beta_tmp)
    plt.xlim(xmin=-10, xmax=30)
    plt.ylim(ymin=0., ymax=1.0)
    plt.plot(global_snr, alpha.T)
    plt.plot(global_snr, np.ones_like(global_snr) * 0.5, 'k:')
    plt.xlabel('global snr of the noisy utterence in training(dB)')
    plt.ylabel('speech distortion weight(alpha)')
    plt.legend(['b=1dB', 'b=5dB', 'b=10dB', 'b=20dB'])
    plt.show()
    print('sc')


def speech_distortion_test():
    main_path = '/Users/audio_source/GaGNet/First_DNS_no_reverb'
    file_name = 'clnsp1_train_69005_1_snr15_tl-21_fileid_158.wav'
    clean_name = os.path.join(main_path, 'no_reverb_clean',
                              'clean_fileid_158.wav')
    noise_name = os.path.join(main_path, 'no_reverb_mix', file_name)
    clean, _ = lib.load(clean_name, sr=16000)
    noisy, _ = lib.load(noise_name, sr=16000)
    noise = noisy - clean

    win_len = 512
    hop_len = 128
    n_fft = 512
    stft_clean = lib.stft(clean,
                          win_length=win_len,
                          hop_length=hop_len,
                          n_fft=n_fft)
    stft_noise = lib.stft(noise,
                          win_length=win_len,
                          hop_length=hop_len,
                          n_fft=n_fft)
    stft_noisy = lib.stft(noisy,
                          win_length=win_len,
                          hop_length=hop_len,
                          n_fft=n_fft)
    clean_mag, _ = lib.magphase(stft_clean)
    noise_mag, _ = lib.magphase(stft_noise)
    noisy_mag, noisy_phase = lib.magphase(stft_noisy)
    # plot_mesh(np.log(clean_mag), 'clean_mag')
    # plot_mesh(np.log(noise_mag), 'noise_mag')
    # plt.show()

    snr = clean_mag / (noise_mag + clean_mag)
    # snr = clean_mag / noise_mag
    # snr = clean_mag / noisy_mag
    enhance_noisy = noisy_mag * snr
    enhance_noise = noise_mag * snr
    enhance_clean = clean_mag * snr
    # plot_mesh(np.log(enhance_noisy), 'enhance_noisy')
    # plot_mesh(np.log(enhance_noise), 'enhance_noise')
    # plot_mesh(np.log(enhance_clean), 'enhance_clean')
    # plt.show()

    enhance_noisy_tmp = enhance_noisy * noisy_phase
    enhance_noise_tmp = enhance_noise * noisy_phase
    enhance_clean_tmp = enhance_clean * noisy_phase
    len_noisy = len(noisy)
    enhance_noisy_t = lib.istft(enhance_noisy_tmp,
                                win_length=win_len,
                                hop_length=hop_len,
                                length=len_noisy)
    enhance_clean_t = lib.istft(enhance_clean_tmp,
                                win_length=win_len,
                                hop_length=hop_len,
                                length=len_noisy)
    enhance_noise_t = lib.istft(enhance_noise_tmp,
                                win_length=win_len,
                                hop_length=hop_len,
                                length=len_noisy)
    sf.write("./enhance_noisy_t.wav", enhance_noisy_t, samplerate=16000)
    sf.write("./enhance_clean_t.wav", enhance_clean_t, samplerate=16000)
    sf.write('./enhance_noise_t.wav', enhance_noise_t, samplerate=16000)

    print('sc')


def wg(S: Tensor, X: Tensor, eps: float = 1e-10):
    N = X - S
    SS = S.abs().square()
    NN = N.abs().square()
    return (SS / (SS + NN + eps)).clamp(0, 1)


def irm(S: Tensor, X: Tensor, eps: float = 1e-10):
    N = X - S
    SS_mag = S.abs()
    NN_mag = N.abs()
    return (SS_mag / (SS_mag + NN_mag + eps)).clamp(0, 1)


def iam(S: Tensor, X: Tensor, eps: float = 1e-10):
    SS_mag = S.abs()
    XX_mag = X.abs()
    return (SS_mag / (XX_mag + eps)).clamp(0, 1)


class Stft(nn.Module):
    def __init__(self,
                 n_fft: int,
                 hop: int = None,
                 window: Tensor = None) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop or n_fft // 4
        if window is not None:
            assert window.shape[0] == n_fft
        else:
            window = torch.hann_window(self.n_fft)
        self.w: torch.Tensor
        self.register_buffer("w", window)

    def forward(self, input: Tensor):
        t = input.shape[-1]
        sh = input.shape[:-1]
        out = torch.stft(input.reshape(-1, t),
                         n_fft=self.n_fft,
                         hop_length=self.hop,
                         window=self.w,
                         normalized=True,
                         return_complex=True)
        out = out.view(*sh, *out.shape[-2:])
        return out


class Istft(nn.Module):
    def __init__(self, n_fft_inv: int, hop_inv: int, window_inv: Tensor):
        super().__init__()
        self.n_fft_inv = n_fft_inv
        self.hop_inv = hop_inv
        # self.window_inv = window_inv
        self.w_inv: torch.Tensor
        self.register_buffer("w_inv", window_inv)

    def forward(self, input: Tensor):
        t, f = input.shape[-2:]
        sh = input.shape[:-2]

        out = torch.istft(F.pad(
            input.reshape(-1, t, f).transpose(1, 2), (0, 1)),
                          n_fft=self.n_fft_inv,
                          hop_length=self.hop_inv,
                          window=self.w_inv,
                          normalized=True)
        if input.ndim > 2:
            out = out.view(*sh, out.shape[-1])

        return out


class MultResSpecLoss(nn.Module):
    gamma: Final[float]
    f: Final[float]
    f_complex: Final[List[float]]

    def __init__(self, n_ffts, gamma, factor, f_complex=None):
        super().__init__()
        self.gamma = gamma
        self.f = factor
        self.stfts = nn.ModuleDict(
            {str(n_fft): Stft(n_fft)
             for n_fft in n_ffts})
        if f_complex is None or f_complex == 0:
            self.f_complex = None
        elif isinstance(f_complex, Iterable):
            self.f_complex = list(f_complex)
        else:
            self.f_complex = [f_complex] * len(self.stfts)

    def forward(self, input: Tensor, target: Tensor):
        loss = torch.zeros()
        for i, stft in enumerate(self.stfts.values()):
            Y = stft(input)
            S = stft(target)
            Y_abs = Y.abs()
            S_abs = S.abs()
            if self.gamma != 1:
                Y_abs = Y_abs.clamp_min(1e-12).pow(self.gamma)
                S_abs = S_abs.clamp_min(1e-12).pow(self.gamma)
            loss += F.mse_loss(Y_abs, S_abs) * self.f
            if self.f_complex is not None:
                if self.gamma != 1:
                    Y = Y_abs * torch.exp(1j * angle.apply(Y))
                    S = S_abs * torch.exp(1j * angle.apply(S))
                loss += F.mse_loss(torch.view_as_real(Y),
                                   torch.view_as_real(S)) * self.f_complex[i]
        return loss


class angle(Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backend(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (x, ) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
        return torch.view_as_complex(
            torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1))


if __name__ == "__main__":
    # snr_weight_loss_adjust()
    speech_distortion_test()
