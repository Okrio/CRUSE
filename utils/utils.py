'''
Author: your name
Date: 2022-02-12 16:16:12
LastEditTime: 2022-02-12 16:21:08
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /CRUSE/utils/utils.py
'''
import numpy as np
import torch
# import matplotlib.pyplot as plt


def postfiltering(indata, mask, tao=0.02):
    iam_sin = mask * torch.sin(np.pi * mask / 2)
    iam_pf = (1 + tao) * mask / (1 + tao * mask**2 / (iam_sin**2))

    return indata * iam_pf
