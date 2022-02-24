'''
Author: Okrio
Date: 2022-02-12 11:12:24
LastEditTime: 2022-02-24 23:12:47
LastEditors: Please set LastEditors
Description: loss function
FilePath: /CRUSE/loss_func/loss.py
'''
import torch
# import numpy as np


class loss_func:
    def __init__(self, loss_mode) -> None:
        assert loss_mode in [
            'SI-SNR', 'SS-SNR', 'MSE', 'Normal_MSE', 'CN_MSE', 'D_MSE',
            'WO_MALE', 'C_MSE'
        ], "Loss mode must be one of ***"
        self.loss_mode = loss_mode

    def loss(self, inputs, labels, noisy=None):
        if self.loss_mode == 'SI-SNR':
            return -(sisnr(inputs, labels))
        elif self.loss_mode == 'SS-SNR':
            return 0
        elif self.loss_mode == 'WO_MALE':
            return wo_male(labels, inputs, noisy)
        elif self.loss_mode == 'C_MSE':
            return c_rmse(labels, inputs)
        elif self.loss_mode == 'MSE':
            return rmse(labels, inputs)


def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


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
    # real_ref = ref[:, 0, :, :]
    # imag_ref = ref[:, 1, :, :]

    # real_est = est[:, 0, :, :]
    # imag_est = est[:, 1, :, :]
    err = est - ref
    mse = torch.sum(torch.sqrt(err**2)) / (B * T * F)
    return mse


def cn_rmse(ref, est, unproc=None, eps=1e-8):
    if ref.shape != est.shape:
        raise RuntimeError(
            f"Dimension mismatch when calculate c_mse, {ref.shape} vs {est.shape}"
        )


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
    tmp1 = torch.pow(mag_est, c)
    tmp2 = torch.pow(mag_ref, c)
    tmp3 = tmp1 * torch.cos(phase_ref) + tmp1 * torch.sin(phase_ref) * 1j

    tmp4 = tmp2 * torch.cos(phase_est) + tmp1 * torch.sin(phase_est) * 1j
    tmp5 = tmp3 - tmp4
    tmp5 = torch.abs(tmp5)

    loss1 = (torch.pow(mag_ref, c) - torch.pow(mag_est, c))**2
    loss2 = tmp5**2
    loss = (1 - beta) * torch.sum(loss1) + beta * torch.sum(loss2)
    return loss


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