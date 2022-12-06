import torch
import torch.nn as nn
import torch.nn.functional as F
from spafe.fbank import linear_fbanks
import einops


class STFT(nn.Module):
    def __init__(self, win_len, hop_len, fft_len, win_type) -> None:
        super(STFT, self).__init__()
        self.win, self.hop = win_len, hop_len
        self.nfft = fft_len
        window = {
            "hann": torch.hann_window(win_len),
            "hamm": torch.hamming_window(win_len),
        }
        assert win_type in window.keys()
        self.window = window[win_type]

    def transform(self, inp):
        cspec = torch.stft(inp,
                           self.nfft,
                           self.hop,
                           self.win,
                           self.window.to(inp.device),
                           return_complex=False)
        cspec = einops.rearrange(cspec, "b f t c -> b c f t")
        return cspec

    def inverse(self, real, imag):
        """
        real, imag:BFT
        """
        inp = torch.stack([real, imag], dim=-1)
        return torch.istft(inp, self.nfft, self.hop,
                           self.window.to(real.device))


class ComplexConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 padding=(0, 0),
                 dilation=1,
                 groups=1,
                 casual=True,
                 complex_axis=1):
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = casual
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis

        self.real_conv = nn.Conv2d(self.in_channels,
                                   self.out_channels,
                                   kernel_size,
                                   self.stride,
                                   padding=(self.padding[0], 0),
                                   dilation=self.dilation,
                                   groups=self.groups)
        self.imag_conv = nn.Conv2d(self.in_channels,
                                   self.out_channels,
                                   kernel_size,
                                   self.stride,
                                   padding=(self.padding[0], 0),
                                   dilation=self.dilation,
                                   groups=self.groups)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.normal_(self.real_conv.bias, 0.)
        nn.init.normal_(self.imag_conv.bias, 0.)

    def forward(self, inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])  # causal
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(real, )
            imag2imag = self.imag_conv(imag)
            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real

        out = torch.cat((real, imag), self.complex_axis)

        return out


def complex_cat(inps, dim=1):
    reals, imags = [], []

    for inp in inps:
        real, imag = inp.chunk(2, dim)
        reals.append(real)
        imags.append(imag)

    reals = torch.cat(reals, dim)
    imags = torch.cat(imags, dim)
    return reals, imags


class ComplexLinearProjection(nn.Module):
    def __init__(self, cin):
        super(ComplexLinearProjection, self).__init__()
        self.clp = ComplexConv2d(cin, cin)

    def forward(self, real, imag):
        """
        real, imag: BCFT
        """

        inputs = torch.cat((real, imag), 1)
        outputs = self.clp(inputs)

        real, imag = outputs.chunk(2, dim=1)
        outputs = torch.sqrt(real**2 + imag**2 + 1e-8)
        return outputs


class PhaseEncoder(nn.Module):
    def __init__(self, cout, n_sig, cin=2, alpha=0.5):
        super(PhaseEncoder, self).__init__()
        self.complexnn = nn.ModuleList()

        for _ in range(n_sig):
            self.complexnn.append(
                nn.Sequential(nn.ConstantPad2d((2, 0, 0, 0), 0.0),
                              ComplexConv2d(cin, cout, (1, 3))))
        self.clp = ComplexLinearProjection(cout * n_sig)
        self.alpha = alpha

    def forward(self, cspecs):
        """
        cspecs: BCFT
        """
        outs = []
        for idx, layer in enumerate(self.complexnn):
            outs.append(layer(cspecs[idx]))
        real, imag = complex_cat(outs, dim=1)

        amp = self.clp(real, imag)
        return amp**self.alpha


class TFCM_Block(nn.Module):
    def __init__(self, cin=24, K=(3, 3), dila=1, causal=True):
        super(TFCM_Block, self).__init__()
        self.pconv1 = nn.Sequential(nn.Conv2d(cin, cin, kernel_size=(1, 1)),
                                    nn.BatchNorm2d(cin), nn.PReLU(cin))
        dila_pad = dila * (K[1] - 1)
        if causal:
            self.dila_conv = nn.Sequential(
                nn.ConstantPad2d((dila_pad, 0, 1, 1), 0.0),
                nn.Conv2d(cin, cin, K, 1, dilation=(1, dila), groups=cin),
                nn.BatchNorm2d(cin), nn.PReLU(cin))
        else:
            self.dila_conv = nn.Sequential(
                nn.ConstantPad2d((dila_pad // 2, dila_pad // 2, 1, 1), 0, 0),
                nn.Conv2d(cin, cin, K, 1, dilation=(1, dila)),
                nn.BatchNorm2d(cin), nn.PReLU(cin))
        self.pconv2 = nn.Conv2d(cin, cin, kernel_size=(1, 1))
        self.causal = causal
        self.dila_pad = dila_pad

    def forward(self, inps):
        """
        inp : BCFT 
        """
        outs = self.pconv1(inps)
        outs = self.dila_conv(outs)
        outs = self.pconv2(outs)
        return outs + inps


class TFCM(nn.Module):
    def __init__(self, cin=24, K=(3, 3), tfcm_layer=6, causal=True) -> None:
        super(TFCM).__init__()
        self.tfcm = nn.ModuleList()
        for idx in range(tfcm_layer):
            self.tfcm.append(TFCM_Block(cin, K, 2**idx, causal=causal))

    def forward(self, inp):
        out = inp
        for idx in range(len(self.tfcm)):
            out = self.tfcm[idx](out)
        return out


class Banks(nn.Module):
    def __init__(self,
                 nfilters,
                 nfft,
                 fs,
                 low_freq=None,
                 high_freq=None,
                 learnable=False) -> None:
        super(Banks, self).__init__()
        self.nfilters, self.nfft, self.fs = nfilters, nfft, fs
        filter = linear_fbanks.linear_filter_banks(nfilts=self.nfilters,
                                                   nfft=self.nfft,
                                                   low_freq=low_freq,
                                                   high_freq=high_freq,
                                                   fs=self.fs)
        filter = torch.from_numpy(filter).float()
        if not learnable:
            self.register_buffer('filter', filter * 1.3)
            self.register_buffer('filter_inv', torch.pinverse(filter))

        else:
            self.filter = nn.Parameter(filter)
            self.filter_inv = nn.Parameter(torch.pinverse(filter))

    def amp2bank(self, amp):
        amp_feature = torch.einsum('bcft,kf->bckt', amp, self.filter)
        return amp_feature

    def bank2amp(self, inputs):
        return torch.einsum("bckt,kt->bcft", inputs, self.filter_inv)


def test_bank():
    import soundfile as sf
    import numpy as np

    stft = STFT(32 * 48, 8 * 48, 32 * 48, "hann")
    net = Banks(256, 32 * 48, 48000)

    sig_raw, sr = sf.read(".wav")
    sig = torch.from_numpy(sig_raw)[None, :].float()
    cspec = stft.transform(sig)

    mag = torch.norm(cspec, dim=1)
    phase = torch.atan2(cspec[:, 1, :, :], cspec[:, 0, :, :])
    mag = mag.unsqueeze(dim=1)
    outs = net.amp2bank(mag)

    outs = net.bank2amp(outs)
    print(F.mse_loss(outs, mag))

    outs = outs.squeeze(dim=1)
    real = outs * torch.cos(phase)
    imag = outs * torch.sin(phase)

    sig_rec = stft.inverse(real, imag)
    sig_rec = sig_rec.cpu().data.numpy()[0]
    min_len = min(len(sig_rec), len(sig_raw))
    sf.write('rs.wav', np.stack([sig_rec[:min_len], sig_raw[:min_len]],
                                axis=1), sr)
    print(np.mean(np.square(sig_rec[:min_len] - sig_raw[:min_len])))


def test_tfcm():
    nnet = TFCM(24)
    inp = torch.randn(2, 24, 256, 101)
    out = nnet(inp)
    print(out.shape)


if __name__ == "__main__":
    net = PhaseEncoder(cout=4, n_sig=1)
    inps = torch.randn(3, 2, 769, 126)
    outs = net([inps])
    print(outs.shape)
    print('sc')
