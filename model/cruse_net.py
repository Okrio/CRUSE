'''
Author: your name
Date: 2022-02-13 19:35:14
LastEditTime: 2022-02-13 23:22:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /CRUSE/model/cruse_net.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class GGRU(nn.Module):
    def __init__(self,
                 in_features=None,
                 out_features=None,
                 mid_features=None,
                 hidden_size=1024,
                 groups=2):
        super(GGRU, self).__init__()
        hidden_size_t = hidden_size // groups
        self.gru_list1 = nn.ModuleList([
            nn.GRU(hidden_size_t, hidden_size_t, 1, batch_first=True)
            for i in range(groups)
        ])

        self.gru_list2 = nn.ModuleList([
            nn.GRU(hidden_size_t, hidden_size_t, 1, batch_first=True)
            for i in range(groups)
        ])
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.groups = groups
        self.mid_features = mid_features

    def forward(self, x):
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.stack(
            [self.gru_list1[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat(
            [self.gru_list2[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = self.ln2(out)

        out = self.view(out.size(0), out.size(1), x.size(1), -1).contiguous()
        out = out.transpose(1, 2).contiguous()
        return out


class unet1(nn.Modules):
    def __init__(
        self,
        in_features,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    ) -> None:
        super(unet1, self).__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
        )

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.conv3 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)

        self.gru = GGRU(groups=2)

        self.conv2_t_1 = nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride)

        self.conv1_t_1 = nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride)
        self.fc = nn.Linear(in_features=in_features, out_features=in_features)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.bn2_t_1 = nn.BatchNorm2d(in_channels)
        self.bn1_t_1 = nn.BatchNorm2d(in_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        out = x
        e1 = self.elu
        pass
