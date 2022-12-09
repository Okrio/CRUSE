import numpy as np
import torch
from torch import Tensor
import colorful

colortool = colorful
colortool.use_style("solarized")


def freq2erb(freq_hz):
    return 9.265 * torch.log1p(freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
    return 24.7 * 9.265 * (torch.exp(n_erb / 9.265) - 1.)


def erb_fb(sr, fft_size, nb_bands, min_nb_freqs):
    """
    slice frequency band to erb, which is not overlap
    """
    nyq_freq = sr / 2
    freq_width = sr / fft_size
    erb_low = freq2erb(torch.Tensor([0.]))
    erb_high = freq2erb(torch.Tensor([nyq_freq]))
    erb = torch.zeros([nb_bands], dtype=torch.int16)
    step = (erb_high - erb_low) / nb_bands
    prev_freq = 0
    freq_over = 0
    for i in range(nb_bands):
        f = erb2freq(erb_low + (i + 1) * step)
        fb = int(torch.round(f / freq_width))
        nb_freqs = fb - prev_freq - freq_over
        if nb_freqs < min_nb_freqs:
            freq_over = min_nb_freqs - nb_freqs
            nb_freqs = min_nb_freqs
        else:
            freq_over = 0
        erb[i] = nb_freqs
        prev_freq = fb
    erb[nb_bands - 1] += 1
    too_large = torch.sum(erb) - (fft_size / 2 + 1)
    if too_large > 0:
        erb[nb_bands - 1] -= too_large
    assert torch.sum(erb) == (fft_size / 2 + 1)

    return erb


def erb_fb_use(width: np.ndarray,
               sr: int,
               normalized: bool = True,
               inverse: bool = False) -> Tensor:
    """
    construct freq2erb transform matrix
    """
    n_freqs = int(np.sum(width))
    all_freqs = torch.linspace(0, sr // 2, n_freqs + 1)[:-1]
    b_pts = np.cumsum([0] + width.tolist()).astype(int)[:-1]
    fb = torch.zeros((all_freqs.shape[0], b_pts.shape[0]))
    for i, (b, w) in enumerate(zip(b_pts.tolist(), width.tolist())):
        fb[b:b + w, i] = 1
    if inverse:
        fb = fb.t()
        if not normalized:
            fb /= fb.sum(dim=1, keepdim=True)
    else:
        if normalized:
            fb /= fb.sum(dim=0)
    return fb


def compute_band_corr(out, x, p, erb_fb):
    bcsum = 0
    for i, (band_size, out_b) in enumerate(zip(erb_fb, out)):
        k = 1. / band_size
        for j in range(band_size):
            idx = bcsum + j
            out_b += (x[idx].real * p[idx].real +
                      x[idx].imag * p[idx].imag) * k
        bcsum += band_size


def band_mean_norm_freq(xs, xout, state, alpha):
    """
    xs: complex32
    others: f32
    """
    out_state = torch.zeros_like(state)
    out_xout = torch.zeros_like(xout)
    for i, (x, s, xo) in enumerate(zip(xs, state, xout)):
        xabs = torch.norm(x)
        out_state[i] = xabs * (1. - alpha) + s * alpha
        out_xout[i] = xabs - out_state[i]


def band_mean_norm_erb(xs, state, alpha):
    """
    all: f32
    """
    out_xs = torch.zeros_like(xs)
    out_state = torch.zeros_like(state)
    for i, (x, s) in enumerate(zip(xs, state)):
        out_state[i] = x * (1. - alpha) + alpha * s
        out_xs[i] -= out_state[i]
        out_xs[i] /= 40.


def band_unit_norm(xs, state, alpha):
    """
    xs: complex32
    """
    out_xs = torch.zeros_like(xs)
    out_state = torch.zeros_like(state)
    for i, (x, s) in enumerate(zip(xs, state)):
        out_state[i] = torch.norm(x) * (1. - alpha + s * alpha)
        out_xs[i] /= torch.sqrt(out_state[i])


def apply_interp_band_gain(out, band_e, erb_fb):
    bcsum = 0
    out_out = torch.zeros_like(out)
    for i, (band_size, b) in enumerate(zip(erb_fb, band_e)):
        for j in range(band_size):
            idx = bcsum + j
            out_out[idx] = out[idx] * b
        bcsum += band_size
    return out_out


def interp_band_gain(out, band_e, erb_fb):
    bcsum = 0
    for i, (band_size, b) in enumerate(zip(erb_fb, band_e)):
        for j in range(band_size):
            idx = bcsum + j
            out[idx] = b
        bcsum += band_size


def apply_band_gain(out, band_e, erb_fb):
    bcsum = 0
    out_out = torch.zeros_like(out)
    for i, (band_size, b) in enumerate(zip(erb_fb, band_e)):
        for j in range(0, band_size):
            idx = bcsum + j
            out_out[idx] = out[idx] * b  # NOTE
        bcsum += band_size
    return out_out


def post_filter(gain):
    beta = 0.02
    eps = 1e-12 * torch.ones_like(gain)
    pi = torch.pi
    g_sin = torch.zeros_like(gain)
    out_gain = torch.zeros_like(gain)
    g_sin = torch.maximum(gain * torch.sin(pi / 2. * gain), eps)
    out_gain = (1.0 + beta) * gain / (1.0 + beta * torch.pow(gain / g_sin, 2))
    return out_gain


def test_erb_fb():

    sr = 48000
    fft_size = 960
    nb_bands = 34
    min_nb_freqs = 3  # 2

    erb = erb_fb(sr=sr,
                 fft_size=fft_size,
                 nb_bands=nb_bands,
                 min_nb_freqs=min_nb_freqs)
    print(colortool.red(f"erb.shape:{erb.shape}"))
    print(colortool.yellow("sc"))


def test_erb_fb_use():
    sr = 48000
    fft_size = 960
    nb_bands = 32
    min_nb_freqs = 2
    erb = erb_fb(sr, fft_size, nb_bands, min_nb_freqs)
    erb = erb.numpy().astype(int)
    fb = erb_fb_use(erb, sr)
    fb_inverse = erb_fb_use(erb, sr, inverse=True)
    print(colortool.red(f"fb:{fb.shape} {fb}"))
    print(colortool.yellow(f"fb_inverse:{fb_inverse.shape}, {fb_inverse}"))
    print('sc')


def test_apply_band_gain():
    """
    test erb band gain for data
    """
    # import random
    sr = 24000
    fft_size = 192
    n_freqs = fft_size // 2 + 1
    nb_bands = 24
    min_nb_freqs = 1
    erb = erb_fb(sr, fft_size, nb_bands, min_nb_freqs)
    # band_e = torch.randint(0, 10, erb.shape)
    mask = torch.ones(nb_bands)
    mask[3] = 0.3
    mask[nb_bands - 1] = 0.5
    input_real = torch.rand(n_freqs)
    input_imag = torch.rand(n_freqs)
    input = torch.complex(input_real, input_imag)
    out_out = apply_band_gain(input, mask, erb)

    out_out1 = torch.zeros_like(out_out)
    cumsum = 0
    for erb_idx, erb_w in enumerate(erb):
        for i in range(cumsum, cumsum + erb_w):
            out_out1[i] = input[i] * mask[erb_idx]
        cumsum += erb_w

    print('sc')


if __name__ == "__main__":
    test_apply_band_gain()
    # test_erb_fb()
    # test_erb_fb_use()
    a = [1, 4, 5]
    b = ['sf', 'fs', 'e']
    for i, (b, w) in enumerate(zip(a, b)):
        print(b, w)
        print('sc')
