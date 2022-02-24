'''
Author: your name
Date: 2022-02-24 22:29:22
LastEditTime: 2022-02-24 22:41:19
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /CRUSE/model/deep_filter.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFilter(nn.Module):
    def __init__(self, t_dim, f_dim):
        super(DeepFilter, self).__init__()
        self.t_dim = t_dim
        self.f_dim = f_dim
        t_width = t_dim * 2 + 1
        f_width = f_dim * 2 + 1
        kernel = torch.eye(t_width * f_width)
        # self.kernel = kernel
        self.register_buffer(
            'kernel',
            torch.reshape(kernel, [t_width(f_width, 1, f_width, t_width)]))

    def forward(self, inputs, filters):
        chunked_inputs = F.conv2d(torch.cat(inputs, 0)[:, None],
                                  self.kernel,
                                  padding=[self.f_dim, self.t_dim])
        inputs_r, inputs_i = torch.chunk(chunked_inputs, 2, 0)
        chunked_filters = F.conv2d(torch.cat(filters, 0)[:, None],
                                   self.kernel,
                                   padding=[self.f_dim, self.t_dim])
        filters_r, filters_i = torch.chunk(chunked_filters, 2, 0)
        outputs_r = inputs_r * filters_r - inputs_i * filters_i
        output_i = inputs_r * filters_i + inputs_r * filters_i
        outputs_r = torch.sum(outputs_r, 1)
        output_i = torch.sum(output_i, 1)
        return torch.cat([outputs_r, output_i], dim=1)


if __name__ == "__main__":
    inputs = [torch.randn(10, 256, 100), torch.randn(10, 256, 100)]
    mask = [torch.randn(10, 256, 100), torch.randn(10, 256, 100)]
    net = DeepFilter(1, 5)
    outputs = net(inputs, mask)
    print(outputs.shape)
    print('sc')
