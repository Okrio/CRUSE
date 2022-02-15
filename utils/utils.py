'''
Author: your name
Date: 2022-02-12 16:16:12
LastEditTime: 2022-02-15 23:28:31
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /CRUSE/utils/utils.py
'''
import numpy as np
import torch
# import matplotlib.pyplot as plt

EPS = np.finfo(float).eps


def normalize(audio, target_level=-25):
    rms = (audio**2).mean()**0.5
    scalar = 10**(target_level / 20) / (rms + 1e-8)
    audio = audio * scalar
    return audio


def active_rms(audio, sr=16000, energy_thresh=-120):
    """
    signal active rms 
    """
    window_size = 100
    window_sample = int(sr * window_size / 1000)
    sample_start = 0
    audio_active_segs = []
    EPS = np.finfo(float).eps

    while sample_start < len(audio):
        sample_end = min(sample_start + window_sample, len(audio))
        audio_win = audio[sample_start:sample_end]
        audio_seg_rms = 10 * np.log10((audio_win**2).mean() + EPS)
        if audio_seg_rms > energy_thresh:
            audio_active_segs = np.append(audio_active_segs, audio_win)
        sample_start += window_sample
    if len(audio_active_segs) != 0:
        audio_rms = (audio_active_segs**2).mean()**0.5
    else:
        audio_rms = EPS
    return audio_rms


def activitydetector(audio, fs=16000, energy_thresh=0.13, target_level=-25):
    audio = normalize(audio, target_level)
    window_size = 50
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0
    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8
    vad_val = np.zeros_like(audio)
    vad_frame_val = []

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 10 * np.log10(sum(audio_win**2) + EPS)
        frame_energy_prob = 1. / (1 + np.exp(-(a + b + frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smooth_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (
                1 - alpha_att)
        else:
            smooth_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (
                1 - alpha_rel)

        if smooth_energy_prob > energy_thresh:
            vad_val[sample_start:sample_end] = 1
            vad_frame_val.append(1)
            active_frames += 1
        else:
            vad_frame_val.append(0)

        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    prec_active = active_frames / cnt
    return prec_active, vad_val, np.array(vad_frame_val)


def activity_detector_amp(audio, fs=16000, thre=9):
    """
    rnnoise vad method

    """
    pass


def write_log_file(log_dir, log_filename, data):
    pass


def postfiltering(indata, mask, tao=0.02):
    iam_sin = mask * torch.sin(np.pi * mask / 2)
    iam_pf = (1 + tao) * mask / (1 + tao * mask**2 / (iam_sin**2))

    return indata * iam_pf
