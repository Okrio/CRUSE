import math
from collections import OrderedDict

from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from typing_extensions import Final


class Conv2dNormAct(nn.Sequential):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 fstride=1,
                 dilation=1,
                 fpad=True,
                 bias=True,
                 separable=False,
                 norm_layer=torch.nn.BatchNorm2d,
                 activation_layer=torch.nn.ReLU):
        """
        [B C T F]
        """
        lookahead = 0
        kernel_size = ((kernel_size, kernel_size) if isinstance(
            kernel_size, int) else tuple(kernel_size))

        if fpad:
            fpad_ = kernel_size[1] // 2 + dilation - 1
        else:
            fpad_ = 0
        pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
        layers = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        if max(kernel_size) == 1:
            separable = False
        layers.append(
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size,
                      padding=(0, fpad_),
                      stride=(1, fstride),
                      dilation=(1, dilation),
                      groups=groups,
                      bias=bias))
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class ConvTranspose2dNormAct(nn.Sequential):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 fstride=1,
                 dilation=1,
                 fpad=True,
                 bias=True,
                 separable=False,
                 norm_layer=torch.nn.BatchNorm2d,
                 activation_layer=torch.nn.ReLU):
        lookahead = 0
        kernel_size = (kernel_size, kernel_size) if isinstance(
            kernel_size, int) else kernel_size
        if fpad:
            fpad_ = kernel_size[1] // 2
        else:
            fpad_ = 0

        pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
        layers = []

        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1

        if groups == 1:
            separable = False
        layers.append(
            nn.ConvTranspose2d(in_ch,
                               out_ch,
                               kernel_size=kernel_size,
                               padding=(kernel_size[0] - 1,
                                        fpad_ + dilation - 1),
                               output_padding=(0, fpad_),
                               stride=(1, fstride),
                               dilation=(1, dilation),
                               groups=groups,
                               bias=bias))
        if separable:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


def convkxf(in_ch,
            out_ch,
            k=1,
            f=3,
            fstride=2,
            lookahead=0,
            batch_norm=False,
            act=torch.nn.ReLU(inplace=True),
            mode="normal",
            depthwise=True,
            complex_in=False):
    bias = batch_norm is False
    assert f % 2 == 1
    stride = 1 if f == 1 else (1, fstride)
    if out_ch is None:
        out_ch = in_ch * 2 if mode == "normal" else in_ch // 2
    fpad = (f - 1) // 2
    convpad = (0, fpad)

    modules = []
    pad = [0, 0, k - 1 - lookahead, lookahead]
    if any(p > 0 for p in pad):
        modules.append(("pad", nn.ConstantPad2d(pad, 0.0)))
    if depthwise:
        groups = min(in_ch, out_ch)
    else:
        groups = 1
    if in_ch % groups != 0 or out_ch % groups != 0:
        groups = 1
    if complex_in and groups % 2 == 0:
        groups //= 2

    convkwargs = {
        "in_channels": in_ch,
        "out_channels": out_ch,
        "kernel_size": (k, f),
        "stride": stride,
        "groups": groups,
        "bias": bias
    }
    if mode == "normal":
        modules.append(("sconv", nn.Conv2d(padding=convpad, **convkwargs)))

    elif mode == "transposed":
        padding = (k - 1, fpad)
        modules.append(("sconv",
                        nn.ConvTranspose2d(padding=padding,
                                           output_padding=convpad,
                                           **convkwargs)))
    elif mode == "upsample":
        modules.append(("upsample", FreqUpsample(fstride)))
        convkwargs["stride"] = 1
        modules.append(("sconv", nn.Conv2d(padding=convpad, **convkwargs)))
    else:
        raise NotImplementedError()
    if groups > 1:
        modules.append(("1x1conv", nn.Conv2d(out_ch, out_ch, 1, bias=False)))
    if batch_norm:
        modules.append(("norm", nn.BatchNorm2d(out_ch)))
    modules.append(("act", act))
    return nn.Sequential(OrderedDict(modules))


class FreqUpsample(nn.Module):
    def __init__(self, factor, mode="nearest"):
        super().__init__()
        self.f = float(factor)
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=(1., self.f), mode=self.mode)


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


class GroupedGRULayer(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    out_size: Final[int]
    bidirectional: Final[bool]
    num_directions: Final[int]
    groups: Final[int]
    batch_first: Final[bool]

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 groups: int,
                 batch_first: bool = True,
                 bias=True,
                 dropout: float = 0,
                 bidirectional=False):
        super().__init__()

        assert input_size % groups == 0
        assert hidden_size % groups == 0
        kwargs = {
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.out_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.groups = groups
        self.batch_first = batch_first
        assert (self.hidden_size %
                groups) == 0, "Hidden size must be divisible by groups"
        self.layers = nn.ModuleList((nn.GRU(self.input_size, self.hidden_size,
                                            **kwargs) for _ in range(groups)))

    def flatten_parameters(self):
        for layer in self.layers:
            layer.flatten_parameters()

    def get_h0(self,
               batch_size: int = 1,
               device: torch.device = torch.device("cpu")):
        return torch.zeros(
            self.groups * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )

    def forward(self,
                input: Tensor,
                h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # input shape: [B, T, I] if batch_first else [T, B, I], B: batch_size, I: input_size
        # state shape: [G*D, B, H], where G: groups, D: num_directions, H: hidden_size

        if h0 is None:
            dim0, dim1 = input.shape[:2]
            bs = dim0 if self.batch_first else dim1
            h0 = self.get_h0(bs, device=input.device)
        outputs: List[Tensor] = []
        outstates: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            o, s = layer(
                input[..., i * self.input_size:(i + 1) * self.input_size],
                h0[i * self.num_directions:(i + 1) *
                   self.num_directions].detach(),
            )
            outputs.append(o)
            outstates.append(s)
        output = torch.cat(outputs, dim=-1)
        h = torch.cat(outstates, dim=0)
        return output, h


class GroupGRU(nn.Module):
    groups: Final[int]
    num_layers: Final[int]
    batch_first: Final[bool]
    hidden_size: Final[int]
    bidirectional: Final[bool]
    num_directions: Final[int]
    shuffle: Final[int]
    add_outputs: Final[bool]

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 groups=4,
                 bias=True,
                 batch_first=True,
                 dropout=0.,
                 bidirectional=False,
                 shuffle=True,
                 add_outputs=False):
        super().__init__()
        kwargs = {
            "groups": groups,
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional
        }
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        assert num_layers > 0
        self.input_size = input_size
        self.groups = groups
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size // groups
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if self.groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.add_outputs = add_outputs
        self.grus: List[GroupedGRULayer] = nn.ModuleList()
        self.grus.append(GroupedGRULayer(input_size, hidden_size, **kwargs))
        for _ in range(1, num_layers):
            self.grus.append(
                GroupedGRULayer(hidden_size, hidden_size, **kwargs))
        self.flatten_parameters()

    def flatten_parameters(self):
        for gru in self.grus:
            gru.flatten_parameters()

    def get_h0(
        self,
        batch_size: int,
    ) -> Tensor:
        return torch.zeros(
            (self.num_layers * self.groups * self.num_directions, batch_size,
             self.hidden_size), )

    def forward(self,
                input: Tensor,
                state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        dim0, dim1, _ = input.shape
        b = dim0 if self.batch_first else dim1
        if state is None:
            state = self.get_h0(b, input.device)
        output = torch.zeros(dim0,
                             dim1,
                             self.hidden_size * self.num_directions *
                             self.groups,
                             device=input.device)
        outstates = []
        h = self.groups * self.num_directions
        for i, gru in enumerate(self.grus):
            input, s = gru(input, state[i * h:(i + 1) * h])
            outstates.append(s)
            if self.shuffle and i < self.num_layers - 1:
                input = (input.view(dim0, dim1, -1, self.groups).transpose(
                    2, 3).reshape(dim0, dim1, -1))
            if self.add_outputs:
                output += input
            else:
                output = input
        outstate = torch.cat(outstates, dim=0)
        return output, outstate


class SqueezedGRU(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip_op: Optional[Callable[..., torch.nn.Module]] = None,
        linear_act_layer: Callable[..., torch.nn.Module] = nn.Identity,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(
            GroupedLinearEinsum(input_size, hidden_size, linear_groups),
            linear_act_layer())
        self.gru = nn.GRU(hidden_size,
                          hidden_size,
                          num_layers=num_layers,
                          batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups),
                linear_act_layer())
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h=None) -> Tuple[Tensor, Tensor]:
        input = self.linear_in(input)
        x, h = self.gru(input, h)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        x = self.linear_out(x)
        return x, h


class SqueezedGRU_S(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip_op: Optional[Callable[..., torch.nn.Module]] = None,
        linear_act_layer: Callable[..., torch.nn.Module] = nn.Identity,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(
            GroupedLinearEinsum(input_size, hidden_size, linear_groups),
            linear_act_layer())
        self.gru = nn.GRU(hidden_size,
                          hidden_size,
                          num_layers=num_layers,
                          batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups),
                linear_act_layer())
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h=None) -> Tuple[Tensor, Tensor]:
        x = self.linear_in(input)
        x, h = self.gru(x, h)
        x = self.linear_out(x)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        return x, h


class GroupedLinearEinsum(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    groups: Final[int]

    def __init__(self, input_size: int, hidden_size: int, groups: int = 1):
        super().__init__()
        # self.weight: Tensor
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        assert input_size % groups == 0, f"Input size {input_size} not divisible by {groups}"
        assert hidden_size % groups == 0, f"Hidden size {hidden_size} not divisible by {groups}"
        self.ws = input_size // groups
        self.register_parameter(
            "weight",
            Parameter(torch.zeros(groups, input_size // groups,
                                  hidden_size // groups),
                      requires_grad=True),
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., I]
        b, t, _ = x.shape
        # new_shape = list(x.shape)[:-1] + [self.groups, self.ws]
        new_shape = (b, t, self.groups, self.ws)
        x = x.view(new_shape)
        # The better way, but not supported by torchscript
        # x = x.unflatten(-1, (self.groups, self.ws))  # [..., G, I/G]
        x = torch.einsum("btgi,gih->btgh", x, self.weight)  # [..., G, H/G]
        x = x.flatten(2, 3)  # [B, T, H]
        return x

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(input_size: {self.input_size}, hidden_size: {self.hidden_size}, groups: {self.groups})"


class GroupedLinear(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    groups: Final[int]
    shuffle: Final[bool]

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 groups: int = 1,
                 shuffle: bool = True):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.groups = groups
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        if groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.layers = nn.ModuleList(
            nn.Linear(self.input_size, self.hidden_size)
            for _ in range(groups))

    def forward(self, x: Tensor) -> Tensor:
        outputs: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            outputs.append(
                layer(x[..., i * self.input_size:(i + 1) * self.input_size]))
        output = torch.cat(outputs, dim=-1)
        if self.shuffle:
            orig_shape = output.shape
            output = (output.view(-1, self.hidden_size, self.groups).transpose(
                -1, -2).reshape(orig_shape))
        return output


def test_grouped_gru():

    g = 2
    h = 4
    i = 2
    b = 1
    t = 5

    m = GroupedGRULayer(i, h, g, batch_first=True)

    input = torch.randn((b, t, i))
    h0 = m.get_h0(b)
    assert list(h0.shape) == [g, b, h // g]
    out, hout = m(input, h0)

    num = 2
    m1 = GroupGRU(i, h, num, g, batch_first=True, shuffle=True)
    h0 = m1.get_h0(b)
    out1, hout1 = m1(input, h0)


def test_erb():
    from matplotlib import pyplot as plt
    import colorful
    colortool = colorful
    colortool.use_style("solarized")

    sr = 48000
    fft_size = 960
    nb_bands = 32
    min_nb_freqs = 2
    erb = erb_fb(sr, fft_size, nb_bands, min_nb_freqs)
    erb = erb.numpy().astype(int)
    fb = erb_fb_use(erb, sr, normalized=True)
    fb_inverse = erb_fb_use(erb, sr, normalized=True, inverse=True)
    print(colortool.red(f"fb:{fb.shape} {fb}"))
    print(colortool.yellow(f"fb_inverse:{fb_inverse.shape}, {fb_inverse}"))
    n_freqs = fft_size // 2 + 1
    input = torch.randn((1, 1, 1, n_freqs), dtype=torch.complex64)
    input_abs = input.abs().square()
    # erb_widths = erb
    py_erb = torch.matmul(input_abs, fb)

    py_out = torch.matmul(py_erb, fb_inverse)
    print(f"py_out:{torch.allclose(input_abs, py_out)}"
          )  # todo[okrio]: erb transform is not equal inverse erb transform

    print(colortool.red(f"py_out:{py_out.shape} "))
    print(colortool)
    plt.figure()
    plt.plot(fb)
    plt.figure()
    plt.plot(fb_inverse.transpose(1, 0))
    print('sc')


if __name__ == "__main__":
    test_erb()
