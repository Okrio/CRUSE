

import torch 
import torchaudio
from torch import nn, Tensor
from typing import Callable, Iterable, List, Optional, Tuple, Union
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from typing_extensions import Final
import math
class GroupedGRULayer(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    out_size: Final[int]
    bidirectional: Final[bool]
    num_directions: Final[int]
    groups: Final[int]
    batch_first: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int,
        batch_first: bool = True,
        bias: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
    ):
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
        assert (self.hidden_size % groups) == 0, "Hidden size must be divisible by groups"
        self.layers = nn.ModuleList(
            (nn.GRU(self.input_size, self.hidden_size, **kwargs) for _ in range(groups))
        )

    def flatten_parameters(self):
        for layer in self.layers:
            layer.flatten_parameters()

    def get_h0(self, batch_size: int = 1, device: torch.device = torch.device("cpu")):
        return torch.zeros(
            self.groups * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )

    def forward(self, input: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
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
                input[..., i * self.input_size : (i + 1) * self.input_size],
                h0[i * self.num_directions : (i + 1) * self.num_directions].detach(),
            )
            outputs.append(o)
            outstates.append(s)
        output = torch.cat(outputs, dim=-1)
        h = torch.cat(outstates, dim=0)
        return output, h


class GroupedGRU(nn.Module):
    groups: Final[int]
    num_layers: Final[int]
    batch_first: Final[bool]
    hidden_size: Final[int]
    bidirectional: Final[bool]
    num_directions: Final[int]
    shuffle: Final[bool]
    add_outputs: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        groups: int = 4,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        shuffle: bool = True,
        add_outputs: bool = False,
    ):
        super().__init__()
        kwargs = {
            "groups": groups,
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
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
        if groups == 1:
            shuffle = False  # Fully connected, no need to shuffle
        self.shuffle = shuffle
        self.add_outputs = add_outputs
        self.grus: List[GroupedGRULayer] = nn.ModuleList()  # type: ignore
        self.grus.append(GroupedGRULayer(input_size, hidden_size, **kwargs))
        for _ in range(1, num_layers):
            self.grus.append(GroupedGRULayer(hidden_size, hidden_size, **kwargs))
        self.flatten_parameters()

    def flatten_parameters(self):
        for gru in self.grus:
            gru.flatten_parameters()

    def get_h0(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Tensor:
        return torch.zeros(
            (self.num_layers * self.groups * self.num_directions, batch_size, self.hidden_size),
            device=device,
        )

    def forward(self, input: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        dim0, dim1, _ = input.shape
        b = dim0 if self.batch_first else dim1
        if state is None:
            state = self.get_h0(b, input.device)
        output = torch.zeros(
            dim0, dim1, self.hidden_size * self.num_directions * self.groups, device=input.device
        )
        outstates = []
        h = self.groups * self.num_directions
        for i, gru in enumerate(self.grus):
            input, s = gru(input, state[i * h : (i + 1) * h])
            outstates.append(s)
            if self.shuffle and i < self.num_layers - 1:
                input = (
                    input.view(dim0, dim1, -1, self.groups).transpose(2, 3).reshape(dim0, dim1, -1)
                )
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
            GroupedLinearEinsum(input_size, hidden_size, linear_groups), linear_act_layer()
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups), linear_act_layer()
            )
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
            GroupedLinearEinsum(input_size, hidden_size, linear_groups), linear_act_layer()
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups), linear_act_layer()
            )
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
            Parameter(
                torch.zeros(groups, input_size // groups, hidden_size // groups), requires_grad=True
            ),
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

    def __init__(self, input_size: int, hidden_size: int, groups: int = 1, shuffle: bool = True):
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
            nn.Linear(self.input_size, self.hidden_size) for _ in range(groups)
        )

    def forward(self, x: Tensor) -> Tensor:
        outputs: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            outputs.append(layer(x[..., i * self.input_size : (i + 1) * self.input_size]))
        output = torch.cat(outputs, dim=-1)
        if self.shuffle:
            orig_shape = output.shape
            output = (
                output.view(-1, self.hidden_size, self.groups).transpose(-1, -2).reshape(orig_shape)
            )
        return output



def test_grouped_gru():
    from icecream import ic

    g = 2  # groups
    h = 4  # hidden_size
    i = 2  # input_size
    b = 1  # batch_size
    t = 5  # time_steps
    m = GroupedGRULayer(i, h, g, batch_first=True)
    m1 = SqueezedGRU(i,h,i,1,g)
    m2 = GroupedGRU(i,h,2,g)
    ic(m)
    ic(m1)
    ic(m2)
    input = torch.randn((b, t, i))
    h0 = m.get_h0(b)
    assert list(h0.shape) == [g, b, h // g]
    out, hout = m(input, h0)
    h0_1 = m2.get_h0(b)
    h1_1 = m

    # Should be exportable as raw nn.Module
    torch.onnx.export(
        m, (input, h0), "out/grouped.onnx", example_outputs=(out, hout), opset_version=13
    )
    # Should be exportable as traced
    m = torch.jit.trace(m, (input, h0))
    torch.onnx.export(
        m, (input, h0), "out/grouped.onnx", example_outputs=(out, hout), opset_version=13
    )
    # and as scripted module
    m = torch.jit.script(m)
    torch.onnx.export(
        m, (input, h0), "out/grouped.onnx", example_outputs=(out, hout), opset_version=13
    )

    # now grouped gru
    num = 2
    m = GroupedGRU(i, h, num, g, batch_first=True, shuffle=True)
    ic(m)
    h0 = m.get_h0(b)
    assert list(h0.shape) == [num * g, b, h // g]
    out, hout = m(input, h0)

    # Should be exportable as traced
    m = torch.jit.trace(m, (input, h0))
    torch.onnx.export(
        m, (input, h0), "out/grouped.onnx", example_outputs=(out, hout), opset_version=13
    )
    # and scripted module
    m = torch.jit.script(m)
    torch.onnx.export(
        m, (input, h0), "out/grouped.onnx", example_outputs=(out, hout), opset_version=13
    )


if __name__ == "__main__":
    test_grouped_gru()
