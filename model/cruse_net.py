'''
Author: your name
Date: 2022-02-13 19:35:14
LastEditTime: 2022-02-13 23:28:52
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
        e1 = self.elu(self.bn1(self.conv1(out)))
        e2 = self.elu(self.bn2(self.conv2(e1)))
        e3 = self.elu(self.bn3(self.conv3(e2)))

        out = e3
        out = self.gru(out)
        out1 = self.elu(self.bn2_t_1(self.conv2_t_1(out)))
        out1 = self.elu(self.bn1_t_1(self.conv2_t_1(out1)))
        out1 = self.fc(out1)

        return out

class unet_2(nn.Module):
    def __init__(self,in_feat=161, ch=(1,8,16,32,64), stride=(1,2), rnn_groups=4):
        super(unet_2,self).__init__()
        self.laynum = len(ch) -1
        hidden_size = (in_feat//2**self.laynum*ch[-1])
        self.ker_x = 2
        self.stride = stride
        self.padding = [self.ker_x -stride[0], 3 - stride[1]]
        for i in range(len(ch)-1):
            setattr(self,"conv{}".format(i+1), nn.Conv2d(ch[i], ch[i+1],(self.ker_x,3), self.stride, self.padding))
            tmp = len(ch) - 1 -i
            setattr(self,"conv{}".format(tmp), nn.Conv2d(ch[tmp-1], ch[i+1],(1,3), self.stride))
            setattr(self,"bn{}".format(i+1), nn.BatchNorm2d(ch[i+1]))
            setattr(self,"bn{}_t".format(i+1), nn.BatchNorm2d(ch[tmp-1]))
            setattr(self,"skip_connect_{}".format(i+1), nn.Conv2d(ch[i+1], ch[i+1],(1,3),bias=False))
        self.gru = GGRU(groups=rnn_groups)
        self.elu = nn.ReLU()
        self.fc = nn.Linear(in_feat, in_feat)
    def forward(self,x):
        out =x
        e1_tmp = self.elu(self.bn1(self.conv1(out)[...,-self.padding[0],:]))
        e2 = self.elu(self.bn2(self.conv2(e1_tmp)[...,-self.padding[0],:]))
        e3 = self.elu(self.bn2(self.conv2(e2)[...,-self.padding[0],:]))
        e4 = self.elu(self.bn2(self.conv2(e3)[...,-self.padding[0],:]))
        skip1 = self.skip_connect_1(e1_tmp)
        skip2 = self.skip_connect_1(e2)
        ski3 = self.skip_connect_1(e3)
        skip4 = self.skip_connect_1(e4)
        b,c,t,f = e3.size()
        out_gru = self.gru(e4)
        out2 = out_gru
        out = out2 + skip4
        d4_1 = self.elu(self.bn4_t(self.conv4_t(out)[...,:-1]))+skip3
        d3_1 = self.elu(self.bn3_t(self.conv3_t(out)[...,:-1]))+skip2
        d2_1 = self.elu(self.bn2_t(self.conv2_t(out)[...,:-1]))+skip1
        d1_1 = nn.Sigmoid()(self.conv1_t(out)[...,:-1])
        return d1_1


if __name__ == "__main__":
    x = torch.randn((2, 1, 10, 161))
    net = unet1()
    y = net(x)
    print(f"{x.shape}->{y.shape}")
