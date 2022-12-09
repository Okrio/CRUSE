'''
Author: Okrio
Date: 2022-02-15 22:21:55
LastEditTime: 2022-03-28 22:53:13
LastEditors: Please set LastEditors
Description: Okrio
FilePath: /CRUSE/dataset/preprocess_dataset.py
'''
from dataclasses import replace
import os
# import sys
import colorful
from pathlib import Path

import librosa
from tqdm import tqdm
import numpy as np
import csv

main_dir_name = "/dahufs/groupdata/codec"
candidate_datasets = [
    os.path.join(main_dir_name,
                 'audio/data_source/audio_classification/audio_noise')
]

candidate_merge_datasets = ['asc_noise_txt', 'dns_2021_noise_train.txt']
dist_file = Path("sns_noise.txt").expanduser().absolute()
dis_merge_file = Path("sns_merge.txt").expanduser().absolute()
low_activity_file = Path("sns_low_activity.txt").expanduser().absolute()

# speech parameters
sr = 16000
wav_in_second = 3
activity_threshold = 0.6
total_hrs = 10000.0
csv_flag = False


def read_csv(csv_name):
    with open(csv_name, 'r') as f:
        tmp = csv.reader(f)
        file_list = []
        for i in tmp:
            file_list.append(i)
    print(os.path.basename(csv_name), ": Total file: ", len(file_list))
    return file_list


def offset_and_limit(data_list, offset, limit):
    data_list = data_list[offset:]
    if limit:
        data_list = data_list[:limit]
    return data_list


def select_specify_file(file_path, ext=''):
    data_list = []
    for i, name in enumerate(file_path):
        for j, type_name in enumerate(ext):
            if type_name in name:
                data_list.append(name)

    return data_list


def multi_txt_file_merge(file_path):
    data_list = []
    for i in file_path:
        data_list += [
            line.rstrip('\n')
            for line in open(os.path.abspath(os.path.expanduser(i)), 'r')
        ]
    with open(dis_merge_file.as_posix(), 'w') as f:
        f.writelines(f"{file_path}\n" for file_path in data_list)
    return data_list


dataset_offset = 0
dataset_limit = None

if __name__ == "__main__":
    # ss = multi_txt_file_merge(candidate_merge_datasets)
    all_wav_path_list = []
    output_wav_path_list = []
    accumulated_time = 0.0

    is_clipped_wav_list = []
    is_low_activity_list = []
    is_too_short_list = []
    is_large_rt60_list = []

    clean_all_wav_path_list = []
    clean_output_wav_path_list = []
    clean_accumulated_time = 0.0

    clean_is_clipped_wav_list = []
    clean_is_low_activity_list = []
    clean_is_too_short_list = []

    rir_is_clipped_wav_list = []
    rir_is_low_activity_list = []
    rir_is_too_short_list = []

    for dataset_path in candidate_datasets:
        if csv_flag:
            dataset_path1 = read_csv(dataset_path)
            dataset_path1 = dataset_path1[1:]
            dataset_path1 = dataset_path1[:]
            all_wav_path_list += dataset_path1
        else:
            dataset_path = Path(dataset_path).expanduser().absolute()
            all_wav_path_list += librosa.util.find_files(
                dataset_path.as_posix(), ext=["wav", "flac"])

    all_wav_path_list = offset_and_limit(all_wav_path_list, dataset_offset,
                                         dataset_limit)

    for wav_file_path in tqdm(all_wav_path_list, desc="Checking..."):
        y, _ = librosa.load(wav_file_path, sr=sr)
        y = y if len(y.shape) > 1 else y[:, np.random.randint(y.shape[-1] + 1)]
        length = np.max(y.shape)
        if length == 0:
            print(wav_file_path)
            continue

        wav_duration = length / sr
        wav_file_user_path = wav_file_path, replace(
            Path(wav_file_path).home().as_posix(), "~")

        is_clipped_wav = 0
        is_low_activity = 0
        is_too_short = 0
        is_large_r60 = 0

        if is_too_short:
            is_too_short_list.append(wav_file_user_path)
            continue
        if is_clipped_wav:
            is_clipped_wav_list.append(wav_file_user_path)
            continue
        if is_low_activity:
            is_low_activity_list.append(wav_file_user_path)
            continue
        if is_large_r60:
            is_large_rt60_list.append(wav_file_user_path)

        if (not is_clipped_wav) and (not is_low_activity) and (
                not is_too_short) and (not is_large_r60):
            accumulated_time += wav_duration
            output_wav_path_list.append(wav_file_user_path)

        if accumulated_time >= (total_hrs * 3600):
            break

    with open(dist_file.as_posix(), "w") as f:
        f.writelines(f"{file_path}\n" for file_path in output_wav_path_list)
    with open(low_activity_file.as_posix(), 'w') as f:
        f.writelines(f"{file_path}\n" for file_path in is_low_activity_list)

    color_spec = colorful
    color_spec.use_style("solarized")

    print("=" * 70)
    print("Speech Preprocessing")
    print(f"\t Original files: {len(all_wav_path_list)}")
    print(
        color_spec.red(
            f"\t Selected files: {accumulated_time / 3600} hrs, {len(output_wav_path_list)} files. "
        ))
    print(f"\t is_clipped_wav: {len(is_clipped_wav_list)}")
    print(f"\t is_low_activity:{len(is_low_activity_list)}")
    print(f"\t is_too_short: {len(is_too_short_list)}")
    print(f"\t is_too_largeRT60: {len(is_large_rt60_list)}")

    print("succeed")
