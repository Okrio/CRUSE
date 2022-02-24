'''
Author: Okrio
Date: 2022-02-17 00:14:37
LastEditTime: 2022-02-25 00:23:53
LastEditors: Please set LastEditors
Description: training dataset generating and validation
FilePath: /CRUSE/dataset/dataset.py
'''

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import signal
import random
from pathlib import Path
import librosa as lib
import soundfile as sf


class BaseDataset(Dataset):
    def __init__(self) -> None:
        super(BaseDataset, self).__init__()

    @staticmethod
    def _offset_and_limit(dataset_list, offset, limit):
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]
        return dataset_list

    @staticmethod
    def _parse_snr_range(snr_range):
        assert len(
            snr_range
        ) == 2, f"The range of snr should be [low, high], not{snr_range}"
        assert snr_range[0] <= snr_range[
            -1], "The low snr should not larger than high snr"

        low, high = snr_range
        snr_list = []
        for i in range(low, high + 1, 1):
            snr_list.append(i)
        return snr_list


class SynDataset(BaseDataset):
    def __init__(self,
                 clean_dataset,
                 clean_dataset_limit,
                 clean_dataset_offset,
                 noise_dataset,
                 noise_dataset_limit,
                 noise_dataset_offset,
                 rir_dataset,
                 rir_dataset_limit,
                 rir_dataset_offset,
                 rir_noise_dataset,
                 rir_noise_dataset_limit,
                 rir_noise_dataset_offset,
                 snr_range,
                 reverb_proportion,
                 reverb_noise_proportion,
                 silence_length,
                 target_dB_FS,
                 target_dB_FS_floating_val,
                 sub_sample_length,
                 sr,
                 dataset_length,
                 pre_load_clean_dataset,
                 pre_load_noise,
                 pre_load_rir,
                 num_workers,
                 valid_mode=False) -> None:
        super(SynDataset, self).__init__()
        self.sr = sr
        self.num_workers = num_workers
        clean_dataset_list = [
            line.rstrip('\n') for line in open(
                os.path.abspath(os.path.expanduser(clean_dataset)), 'r')
        ]
        noise_dataset_list = [
            line.rstrip('\n') for line in open(
                os.path.abspath(os.path.expanduser(noise_dataset)), 'r')
        ]
        rir_dataset_list = [
            line.rstrip('\n') for line in open(
                os.path.abspath(os.path.expanduser(rir_dataset)), 'r')
        ]
        rir_noise_dataset_list = [
            line.rstrip('\n') for line in open(
                os.path.abspath(os.path.expanduser(rir_noise_dataset)), 'r')
        ]

        clean_dataset_list = self._offset_and_limit(clean_dataset_list,
                                                    clean_dataset_offset,
                                                    clean_dataset_limit)
        noise_dataset_list = self._offset_and_limit(noise_dataset_list,
                                                    noise_dataset_offset,
                                                    noise_dataset_limit)
        rir_dataset_list = self._offset_and_limit(rir_dataset_list,
                                                  rir_dataset_offset,
                                                  rir_dataset_limit)
        rir_noise_dataset_list = self._offset_and_limit(
            rir_noise_dataset_list, rir_noise_dataset_limit,
            rir_noise_dataset_offset)

        # if pre_load_clean_dataset:
        #     clean_dataset_list = self._

        self.clean_dataset_list = clean_dataset_list
        self.noise_dataset_list = noise_dataset_list
        self.rir_dataset_list = rir_dataset_list
        self.rir_noise_dataset_list = rir_noise_dataset_list

        self.dataset_length = dataset_length
        self.valid_mode = valid_mode
        snr_list = self._parse_snr_range(snr_range=snr_range)
        self.snr_list = snr_list

        assert 0 <= reverb_proportion <= 1, "reverbation proportion should be in [0,1]"
        self.reverb_proportion = reverb_proportion

        assert 0 <= reverb_noise_proportion <= 1, "reverb_noise proportion should be in [0,1]"
        self.reverb_noise_proportion = reverb_noise_proportion
        self.silence_length = silence_length
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_val = target_dB_FS_floating_val
        self.sub_sample_length = sub_sample_length
        self.length = len(self.clean_dataset_list) if bool(
            self.dataset_length) is False else int(dataset_length)
        self.general_mix_dataset_list = np.random.randint(
            0, len(self.clean_dataset_list), self.length)

    def __len__(self):
        return self.length

    # def _preload_dataset(self,file_path_list, remark=''):
    #     waveform_list = Parallel(n_jobs=self.num_workers)(delayed((l)))

    @staticmethod
    def _random_select_from(dataset_list):
        return random.choice(dataset_list)

    def _select_clean_y(self, sig, target_length):
        sig_len = len(sig)
        clean_y = sig
        silence = np.zeros(int(self.sr * self.silence_length),
                           dtype=np.float32)
        remain_length = target_length - sig_len
        while remain_length > 0:
            clean_file = self._random_select_from(self.clean_dataset_list)
            clean_new_added, _ = lib.load(clean_file, sr=self.sr)
            clean_new_added = clean_new_added if clean_new_added.ndim < 2 else clean_new_added[:,
                                                                                               np
                                                                                               .
                                                                                               random
                                                                                               .
                                                                                               randint(
                                                                                                   clean_new_added
                                                                                                   .
                                                                                                   shape[
                                                                                                       -1]
                                                                                               )]
            clean_y = np.append(clean_y, clean_new_added)
            remain_length = remain_length - len(clean_new_added)

            if remain_length > 0:
                silence_len = min(remain_length, len(silence))
                clean_y = np.append(clean_y, silence[:silence_len])
                remain_length -= silence_len

        if len(clean_y) > target_length:
            idx_start = np.random.randint(len(clean_y) - target_length)
            clean_y = clean_y[idx_start:idx_start + target_length]

        assert len(
            clean_y
        ) == target_length, "the clean_y length equals target_length"
        return clean_y

    def _select_noise_y(self, target_length):
        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(self.sr * self.silence_length),
                           dtype=np.float32)
        remaining_length = target_length
        while remaining_length > 0:
            noise_file = self._random_select_from(self.noise_dataset_list)
            noise_new_added, _ = lib.load(
                noise_file, sr=self.sr)  # todo: check multi-channel wav
            noise_y = np.append(noise_y, noise_new_added)
            remaining_length = remaining_length - len(noise_new_added)
            if remaining_length > 0:
                silence_len = min(remaining_length, len(silence))
                noise_y = np.append(noise_y, silence[:silence_len])
                remaining_length -= silence_len
        if len(noise_y) > target_length:
            idx_start = np.random.randint(len(noise_y) - target_length)
            noise_y = noise_y[idx_start:idx_start + target_length]

        return noise_y

    def _select_rir(self, rir_proportion, rir_dataset_list):
        use_reverb = bool(np.random.random(1) < rir_proportion)
        if use_reverb:
            rir_path = self._random_select_from(rir_dataset_list)
            rir_y, _ = lib.load(
                rir_path, sr=self.sr)  # todo:  check multi-channel rir wav
        else:
            rir_y = None
        return rir_y

    @staticmethod
    def add_reverb(cln_wav, rir_wav, channels=1, predelay=50, sr=16000):
        rir_len = rir_wav.shape[0]
        wav_tgt = np.zeros([channels, cln_wav.shape[0] + rir_len - 1])
        dt = np.argmax(rir_wav, 0).min()
        et = dt + (predelay * sr) // 1000
        et_rir = rir_wav[:et]
        wav_early_tgt = np.zeros(
            [channels, cln_wav.shape[0] + et_rir.shape[0] - 1])
        cln_wav = cln_wav if cln_wav.ndim < 2 else cln_wav[np.random.randint(
            cln_wav.shape[-1])]
        for i in range(channels):
            wav_tgt[i] = signal.fftconvolve(cln_wav, rir_wav[:, i])
            wav_early_tgt[i] = signal.fftconvolve(cln_wav, et_rir[:, i])
        wav_tgt = np.transpose(wav_tgt)
        wav_tgt = wav_tgt[:cln_wav.shape[0]]
        wav_early_tgt = np.transpose(wav_early_tgt)
        wav_early_tgt = wav_early_tgt[:cln_wav.shape[0]]
        return wav_tgt, wav_early_tgt

    @staticmethod
    def snr_mix(clean_y,
                noise_y,
                snr,
                target_dB_FS,
                target_dB_FS_floating_val,
                rir=None,
                rir_noise=None,
                eps=1e-7):
        if rir is not None:
            clean_y = signal.fftconvolve(clean_y, rir)[:len(clean_y)]
        if rir_noise is not None:
            noise_y = signal.fftconvolve(noise_y, rir_noise)[:len(noise_y)]

        # todo spectral augmentation procedure

        clean_y = clean_y / (np.max(np.abs(clean_y)) + eps)

        clean_rms = (clean_y**2).mean()**0.5

        noise_y = noise_y / (np.max(np.abs(noise_y)) + eps)

        noise_rms = (noise_y**2).mean()**0.5
        snr_scalar = clean_rms / (10**(snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y

        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_val,
            target_dB_FS + target_dB_FS_floating_val)
