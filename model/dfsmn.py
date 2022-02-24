'''
Author: your name
Date: 2022-02-24 22:06:47
LastEditTime: 2022-02-24 22:27:23
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /CRUSE/model/dfsmn.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFSMN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 left_frames=1,
                 left_dilation=1,
                 right_frames=1,
                 right_dilation=1):
        super(DFSMN, self).__init__()
        self.left_frames = left_frames
        self.right_frames = right_frames
        self.in_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        # self.norm=nn.InstanceNorm1d(hidden_dim)
        # nn.init.normal_(self.in_conv.weight.data, std=0.05)
        if left_frames > 0:
            self.left_conv = nn.Sequential(
                nn.ConstantPad1d([left_dilation * left_frames, 0], 0),
                nn.Conv1d(hidden_dim,
                          hidden_dim,
                          kernel_size=left_frames + 1,
                          dilation=left_dilation,
                          bias=False,
                          groups=hidden_dim))
        # nn.init.normal_(self.left_conv[1].weight.data,std=0.05)
        if right_frames > 0:
            self.right_conv = nn.Sequential(
                nn.ConstantPad1d(
                    [-right_dilation, right_frames * right_dilation], 0),
                nn.Conv1d(hidden_dim,
                          hidden_dim,
                          kernel_size=right_frames,
                          dilation=right_dilation,
                          bias=False,
                          groups=hidden_dim))
        # nn.init.normal_(self.right_conv[1].weight.data,std=0.05)
        self.out_conv = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        # nn.init.normal_(self.out_conv.weight.data,std=0.05)
        self.weight = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, inputs, hidden=None):
        out = self.in_conv(inputs)
        # out = F.relu(out)
        # out = self.norm(out)
        if self.left_frames > 0:
            left = self.left_conv(out)
        else:
            left = 0
        if self.right_frames > 0:
            right = self.right_conv(out)
        else:
            right = 0
        out_p = out + left + right
        if hidden is not None:
            out_p = hidden + F.relu(out_p) * self.weight
        out = self.out_conv(out_p)
        return out, out_p


if __name__ == "__main__":
    inputs = torch.randn(10, 257, 199)
    net = DFSMN(257, 128, 137, left_dilation=2, right_dilation=3)
    print(net(inputs)[0].shape)
    print('sc')
