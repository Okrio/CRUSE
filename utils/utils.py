'''
Author: your name
Date: 2022-02-12 16:16:12
LastEditTime: 2022-03-28 22:53:14
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /CRUSE/utils/utils.py
'''

# from ntpath import join

from pathlib import Path
import numpy as np
# from sklearn.linear_model import LogisticRegressionCV
# from scipy.fft import fft
import torch
# import matplotlib.pyplot as plt
import os
import csv
import glob
import librosa as lib
import statistics as stats
import soundfile as sf
import scipy.signal as scs
import sys
from typing import List
import colorful

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "test")))
print(os.path.abspath(os.path.join(os.getcwd(), "test")))

from test_loss import plot_mesh

EPS = np.finfo(float).eps


def normalize(audio, target_level=-25):
    rms = (audio**2).mean()**0.5
    scalar = 10**(target_level / 20) / (rms + 1e-8)
    audio = audio * scalar
    return audio


def as_windowed(x: torch.Tensor, win_len, hop_len=1, dim=1):
    """
    input: B, T
    output: B, T//win_len, win_len
    """
    shape: List[int] = list(x.shape)
    stride: List[int] = list(x.stride())
    shape[dim] = int((shape[dim] - win_len + hop_len) // hop_len)
    shape.insert(dim + 1, win_len)
    stride.insert(dim + 1, stride[dim])
    stride[dim] = stride[dim] * hop_len
    y = x.as_strided(shape, stride)
    return y


def torch_active_rms(audio: torch.Tensor,
                     sr=16000,
                     thr=-120,
                     frame_length=100):
    window_samples = int((sr * frame_length) // 1000)
    EPS = torch.finfo(torch.float32).eps

    y = as_windowed(audio, window_samples, window_samples)  # stride input data
    audio_seg_rms = 20 * torch.log10(
        torch.mean(y**2, dim=-1, keepdim=True) + EPS)
    thr_mat = torch.zeros_like(y)
    y1 = torch.where(audio_seg_rms > thr, y, thr_mat)
    y1_flatten = torch.flatten(y1,
                               start_dim=1)  # flatten data in batch dimenstion
    y1_zeros_count = (y1_flatten == 0.).sum(
        dim=-1)  # statistic zero numbers for each batch
    y1_flatten_sum = torch.sum(y1_flatten**2, dim=-1)  # calcuate square sum
    y1_flatten_mean = (y1_flatten_sum / (y1_flatten.shape[-1] - y1_zeros_count)
                       )**0.5  # mean data without zero-val part
    out = y1_flatten_mean.unsqueeze(dim=1)
    return out


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


def vad_simplify(audio, win_len=256, hop_len=160, fs=16000, target_level=-25):
    """
    refer to "weighted speech distortion losses for neural network-based real-time speech enhancement"
    """
    audio = normalize(audio, target_level)
    # audio_len = len(audio)
    # n_frames = (audio_len - win_len + hop_len) // hop_len
    # audio_clips = torch.tensor(audio).unfold(0, win_len, hop_len)
    # print(f"audio_clips:{audio_clips.shape}")
    freq_res = fs * 1. / win_len
    f300Hz_point = int(np.floor(300 / freq_res))
    f5000Hz_point = int(np.ceil(5000 / freq_res))

    stft_audio = lib.stft(audio,
                          n_fft=win_len,
                          hop_length=hop_len,
                          win_length=win_len,
                          center=True)
    stft_mag, _ = lib.magphase(stft_audio)  # F * T
    stft_mag_log = 10 * np.log10(stft_mag**2 + 1e-12)
    plot_mesh(stft_mag_log, 'stft_mag_log')
    # plt.show()
    stft_300_5000_sum = np.sum(stft_mag_log[f300Hz_point:f5000Hz_point, :],
                               axis=0)
    stft_300_5000_sum_smooth = scs.lfilter([0.1], [1, -0.5, -0.2, -0.2],
                                           stft_300_5000_sum)
    plt.figure()
    plt.plot(stft_300_5000_sum)
    plt.plot(stft_300_5000_sum_smooth)
    # ax = plt.axes()
    # ax.set_alpha(0.4)
    plt.legend(['stft_sum', 'stft_sum_smooth'])
    plt.show()

    return


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
    window_size = 20  # ms
    window_sample = int(fs * window_size / 1000)
    start = 0
    data_len = len(audio)
    audio = audio * 32768
    frameshift = window_sample // 2
    nframe = (data_len - window_sample + frameshift) // frameshift
    Energy = np.zeros(data_len)
    vad_seq = np.zeros_like(Energy)
    vad1 = 0
    vad_cnt = 0
    for i in range(0, nframe):
        tmp = np.array(audio[start:i + window_sample])
        E_val = np.sum(tmp * tmp)
        if E_val > 1e9:
            vad_cnt = 0
        elif E_val > 1e8:
            vad_cnt = vad_cnt - 5
        elif E_val > 1e7:
            vad_cnt = vad_cnt + 1
        else:
            vad_cnt += 2
        # todo ...
    pass


def activity_detector_tf_frame(audio, fs=16000, thr=9):

    pass


def write_log_file(log_dir, log_filename, data):
    data = zip(*data)
    with open(os.path.join(log_dir, log_filename), mode="w",
              newline='') as csvfile:
        csvwriter = csv.writer(csvfile,
                               delimiter=' ',
                               quotechar='|',
                               quoting=csv.QUOTE_MINIMAL)
        for row in data:
            csvwriter.writerow([row])


def get_dir(cfg, param_name, new_dir_name):
    if param_name in cfg:
        dir_name = cfg[param_name]

    else:
        dir_name = os.path.join(os.path.dirname(__file__), new_dir_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def statist_vad(data_path):
    noise_filename = glob.glob(os.path.join(data_path), '*.wav')
    noise_filename_list = []
    vad_results_list = []
    total_clips = len(noise_filename)
    for noisepath in noise_filename:
        noise_filename_list.append(os.path.basename(noisepath))
        noise_signal, sr_noise = lib.load(noisepath, sr=16000)
        per_act, _ = activitydetector(noise_signal)
        vad_results_list.append(per_act)

    pc_vad_passed = round(
        vad_results_list.count('True') / total_clips * 100, 1)
    print('% noise clips that passed vad tests: ', pc_vad_passed)
    dir_name = os.path.join(os.path.dirname(__file__), 'Unit_tests_logs')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(dir_name):
        dir_name = os.path.join(os.path.dirname(__file__), "Unit_tests_logs")
        os.makedirs(dir_name)
    write_log_file(dir_name, 'unit_test_results.csv',
                   [noise_filename_list, vad_results_list])


def cal_rt60(y):
    freq_third = [
        400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
        6300, 8000, 10000
    ]
    freqbands = [
        355, 447, 562, 708, 891, 1122, 1413, 1778, 2239, 2818, 3548, 4467,
        5623, 7079, 8913, 11220
    ]
    maxlev = 2**15 - 1
    dbscale = 20
    # medavgtime = 0.3
    ratiofmax = 0.7
    convolven = 2500
    rt60raw = [0.0] * len(freq_third)
    sig, sr = lib.load(y)
    sig = sig[0, :] if sig.ndim > 1 else sig  # todo
    da = sig
    for k in range(len(freq_third)):
        daf = np.fft.rfft(da)
        lofreq = round((freqbands[k + 0] / (sr / 2)) * (len(daf) - 1))
        hifreq = round((freqbands[k + 1] / (sr / 2)) * (len(daf) - 1))
        daf[:lofreq] = 0
        daf[:hifreq] = 0
        nda = np.fft.ifft(daf, len(da))
        nda = abs(nda)
        ndalog = [0.0] * len(nda)
        ndapre = [0.0] * len(nda)
        for i in range(len(nda)):
            if nda[i] != 0:
                ndalog[i] = dbscale * np.log10(nda[i] / maxlev)
            else:
                ndalog[i] = dbscale * np.log10(1 / maxlev)
            ndapre[i] = ndalog[i]

        ndalog = np.convolve(ndalog,
                             np.ones((convolven, )) / convolven,
                             mode='valid')
        ndalog_min, ndalog_max = min(ndalog), max(ndalog)
        ndalog_cut_apx = ndalog_max - (ndalog_max - ndalog_min) * ratiofmax
        ndalog_cut_ind = (np.abs(ndalog - ndalog_cut_apx)).argmin()
        ndalog = ndalog[0:ndalog_cut_ind]

        temp_index = np.arange(0, len(ndalog))
        slope, intercept, r_value, p_value, std_err = stats.Linregress(
            temp_index, ndalog)
        # dBlossline = slope * temp_index + intercept
        rt60 = -60.0 / (slope * sr)
        rt60raw[k] = rt60
    # print('rt60_median:{},{}'.format(np.mean(rt60raw), np.median(rt60raw)))
    return rt60raw


def statist_rt60(data_path, savefile_path):
    rir_filename = lib.util.find_files(data_path, ext=["wav"])
    rir_filename_list = []
    rt60_filename_list = []
    total_clips = len(rir_filename)

    for rirpath in rir_filename:
        rir_filename_list.append(rirpath)
        rt60 = cal_rt60(rirpath)
        rt60_filename_list.append(np.median(rt60))

    target_file_name = 'Unit_test_{}_logs'.format(os.path.basename(data_path))
    dir_name = os.path.join(savefile_path, target_file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(dir_name):
        dir_name = os.path.join(savefile_path, target_file_name)
        os.makedirs(dir_name)
    write_log_file(dir_name, 'Unit_test_rt60_results.csv',
                   [rir_filename_list, rt60_filename_list])


def postfiltering(mask, indata=None, tao=0.02):
    iam_sin = mask * np.sin(np.pi * mask / 2)
    iam_pf = (1 + tao) * mask / (1 + tao * mask**2 / (iam_sin**2))

    return iam_pf


def envelope_postfiltering(unproc, mask, tao=0.02):
    """
    perceptually-motivated 
    Note: only for irm, iam cannot work
    """
    g_hat_b_w = mask * np.sin(np.pi * 0.5 * mask)
    e0 = mask * unproc
    e1 = g_hat_b_w * unproc
    tmp = e0 / (e1 + np.finfo(float).eps)
    g = np.sqrt((1 + tao) * tmp / (1 + tao * tmp**2))
    return g * g_hat_b_w


class PreProcess:
    def __init__(self,
                 win_len,
                 win_inc,
                 fft_len,
                 win_type,
                 post_process_mode,
                 loss_mode,
                 use_cuda=False):
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type
        self.post_process_mode = post_process_mode
        self.loss_mode = loss_mode
        self.use_cuda = use_cuda

        if win_type == "hanning":
            self.window = torch.hann_window(self.fft_len)
        else:
            raise ValueError("ERROR window type")
        if use_cuda:
            self.window = self.window.cuda()

    def pre_stft(self, inputs):
        stft_inputs = torch.stft(inputs,
                                 n_fft=self.fft_len,
                                 hop_length=self.win_inc,
                                 win_length=self.win_len,
                                 window=self.window,
                                 center=True,
                                 pad_mode="constant")
        stft_inputs = stft_inputs.transpose(1, 3).contiguous()
        real = stft_inputs[:, 0, :, :]
        imag = stft_inputs[:, 1, :, :]
        spec_mags = torch.sqrt(real**2 + imag**2 + 1e-8)
        spec_phase = torch.atan2(imag, real)

        real = torch.unsqueeze(real, dim=1)
        imag = torch.unsqueeze(imag, dim=1)
        spec_mags = torch.unsqueeze(spec_mags, dim=1)
        spec_phase = torch.unsqueeze(spec_phase, dim=1)

        self.real = real
        self.imag = imag
        self.spec_mags = spec_mags
        self.spec_phase = spec_phase
        return stft_inputs, real, imag, spec_mags, spec_phase

    def log_transform(self):
        self.spec_mags = torch.log(self.spec_mags)

    def masking(self, mask_real, mask_imag=None):
        if self.post_process_mode == "mag_mapping":
            out_real = mask_real * self.real
            out_imag = mask_real * self.imag
        elif self.post_process_mode == "complex_mapping":
            out_real = mask_real * self.real
            out_imag = mask_imag * self.imag
        elif self.post_process_mode == "mapping":
            out_real = mask_real
            out_imag = mask_imag
        else:
            NotImplementedError

        out_real = out_real.squeeze(1)
        out_imag = out_imag.squeeze(1)
        out_spec = torch.stack([out_real, out_imag], dim=-1).contiguous()
        return out_spec

    def refsig_process(self, indatas):
        if self.loss_mode == "freq":
            out, _, _, _, _ = self.pre_stft(indatas)

        elif self.loss_mode == "time":
            out = indatas
        return out

    def reconstruction(self, stft_outputs, sig_len=None):
        if not isinstance(stft_outputs, torch.Tensor):
            stft_outputs = torch.from_numpy(stft_outputs).type(
                torch.FloatTensor)

        estimated_audio = torch.istft(stft_outputs,
                                      n_fft=self.fft_len,
                                      hop_length=self.win_inc,
                                      win_length=self.win_len,
                                      window=self.window,
                                      center=True,
                                      length=sig_len)
        return estimated_audio


def test_torch_activate_rms():
    x = torch.randn(3, 64)
    y = torch_active_rms(audio=x, frame_length=1, thr=0)
    colortool = colorful
    colortool.use_style("solarized")
    print(colortool.red(f'sc: {y.shape}'))


if __name__ == "__main__":
    test_torch_activate_rms()
    import matplotlib.pyplot as plt
    import librosa.display

    wav_path_dir = "/Users/audio_source/GaGNet/First_DNS_no_reverb/"
    clean_wav_folder_name = "no_reverb_clean"
    mix_wav_folder_name = "no_reverb_mix"

    mix_wav_path_name = os.path.join(wav_path_dir, mix_wav_folder_name)
    mix_dataset_path = Path(mix_wav_path_name).expanduser().absolute()
    mix_all_lists = lib.util.find_files(mix_dataset_path.as_posix(),
                                        ext=['wav'],
                                        limit=1)
    # getting clean signle location
    mix_name = os.path.basename(mix_all_lists[0])
    mix_name_split = mix_name.split('_')
    clean_name = 'clean_' + mix_name_split[-2] + '_' + mix_name_split[-1]
    clean_wav_path_name = os.path.join(wav_path_dir, clean_wav_folder_name,
                                       clean_name)

    mix_sig, _ = lib.load(mix_all_lists[0], sr=16000)
    clean_sig, _ = lib.load(clean_wav_path_name, sr=16000)
    noise_sig = mix_sig - clean_sig

    # test vad_simplify
    vad_simplify(mix_sig, win_len=512, hop_len=128)

    mix_sig_fd = lib.stft(mix_sig, n_fft=256, hop_length=160, win_length=256)
    mix_sig_mag_fd, mix_sig_phase_fd = lib.magphase(mix_sig_fd)
    clean_sig_fd = lib.stft(clean_sig,
                            n_fft=256,
                            hop_length=160,
                            win_length=256)
    clean_sig_mag_fd, clean_sig_phase_fd = lib.magphase(clean_sig_fd)

    noise_sig_fd = lib.stft(noise_sig,
                            n_fft=256,
                            hop_length=160,
                            win_length=256)
    noise_sig_mag_fd, noise_sig_phase_fd = lib.magphase(noise_sig_fd)

    iam = clean_sig_mag_fd / mix_sig_mag_fd
    irm = clean_sig_mag_fd / (clean_sig_mag_fd + noise_sig_mag_fd)
    iam_filter_sig = iam * mix_sig_fd
    iam_filter_sig = lib.istft(iam_filter_sig,
                               hop_length=160,
                               win_length=256,
                               length=len(mix_sig))
    ks_filter_sig = postfiltering(mask=iam) * mix_sig_fd
    ks_filter_sig = lib.istft(ks_filter_sig,
                              hop_length=160,
                              win_length=256,
                              length=len(mix_sig))
    g = envelope_postfiltering(unproc=mix_sig_mag_fd, mask=irm)
    amazon_filter_sig = g * mix_sig_fd
    amazon_filter_sig = lib.istft(amazon_filter_sig,
                                  hop_length=160,
                                  win_length=256,
                                  length=len(mix_sig))

    sf.write('/Users/audio_source/result/iam_out.wav',
             iam_filter_sig,
             samplerate=16000)
    sf.write('/Users/audio_source/result/ks_filter_sig_out.wav',
             ks_filter_sig,
             samplerate=16000)
    sf.write('/Users/audio_source/result/amazon_filter_sig.wav',
             amazon_filter_sig,
             samplerate=16000)

    plt.figure(1)

    librosa.display.specshow(lib.amplitude_to_db(clean_sig_fd, ref=np.max),
                             fmax=8000,
                             y_axis='linear',
                             x_axis='time')
    plt.title('clean signal')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    # plt.show()

    plt.figure(2)

    librosa.display.specshow(lib.amplitude_to_db(mix_sig_fd, ref=np.max),
                             fmax=8000,
                             y_axis='linear',
                             x_axis='time')
    plt.title('mix signal')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    indata = np.linspace(0, 1, 1001)
    beta = 0.02
    y = np.sqrt((1 + beta) * indata / (1 + beta * indata**2))
    y = indata * np.sin(np.pi * 0.5 * indata)
    plt.figure()
    plt.plot(indata, y, color='r')
    plt.show()
    print('sc')
