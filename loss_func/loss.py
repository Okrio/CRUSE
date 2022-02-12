'''
Author: your name
Date: 2022-02-12 11:12:24
LastEditTime: 2022-02-12 16:33:52
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /CRUSE/loss_func/loss.py
'''
import torch
import numpy as np


class loss_func():
    def __init__(self) -> None:
        pass


def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)


def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True)
    data = data - mean
    return data


def sisnr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_noise = s1 - s_target
    targer_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)
    snr = 10 * torch.log10((targer_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)


def rmse(ref, est, eps=1e-8):
    """
    ref: BCTF
    est: BCTF

    """
    if ref.shape != est.shape:
        raise RuntimeError(
            f"Dimension mismatch when calculate rmse, {ref.shape} vs {est.shape}"
        )

    B, C, T, F = torch.size(ref)
    real_ref = ref[:, 0, :, :]
    imag_ref = ref[:, 1, :, :]

    real_est = est[:, 0, :, :]
    imag_est = est[:, 1, :, :]
    err = est - ref
    mse = torch.sum(err**2)


def c_rmse(ref, est, unproc=None, eps=1e-8):
    if ref.shape != est.shape:
        raise RuntimeError(
            f"Dimension mismatch when calculate c_mse, {ref.shape} vs {est.shape}"
        )
    c = 0.3
    beta = 0.3
    B, C, T, F = torch.size(ref)
    real_ref = ref[:, 0, :, :]
    imag_ref = ref[:, 1, :, :]

    real_est = est[:, 0, :, :]
    imag_est = est[:, 1, :, :]

    mag_ref = torch.sqrt(real_ref**2 + imag_ref**2)
    phase_ref = torch.atan2(imag_ref, real_ref)

    mag_est = torch.sqrt(real_est**2 + imag_est**2)
    phase_est = torch.atan2(imag_est, real_est)

    loss1 = (torch.pow(mag_ref, c) - torch.pow(mag_est, c))**2


def wo_male(ref, est, unproc, eps=1e-8):
    if ref.shape != est.shape:
        raise RuntimeError(
            f"Dimension mismatch when calculate wo-male, {ref.shape} vs {est.shape}"
        )
    alpha = 2
    beta = 1
    gamma = 1
    B, C, T, F = torch.size(ref)
    real_ref = ref[:, 0, :, :]
    imag_ref = ref[:, 1, :, :]

    real_est = est[:, 0, :, :]
    imag_est = est[:, 1, :, :]
    mag_ref = torch.sqrt(real_ref**2 + imag_ref**2)
    mag_est = torch.sqrt(real_est**2 + imag_est**2)
    # phase_ref = torch.atan2(imag_ref, real_ref)
    # phase_est = torch.atan2(imag_est, real_est)
    mag_unproc = torch.sqrt(unproc[:, 0, :, :]**2 + unproc[:, 1, :, 1]**2)
    # phase_unproc = torch.atan2(unproc[:, 1, :, :], unproc[:, 0, :, :])

    iam = (mag_ref / mag_unproc)**gamma
    W_iam = torch.exp(alpha / (beta + iam))

    loss = W_iam * torch.abs(
        torch.log10(mag_est + 1) - torch.log10(mag_ref + 1))
    loss = torch.sum(loss) / (B * T * F * 1.0)
    return loss


if __name__ == "__main__":
    x = torch.ones((2, 1, 3, 4))
    y = torch.sum(x, dim=0)
    print(f"y shape:{y.shape}")
    print('sc')