'''
Author: your name
Date: 2022-03-02 22:19:22
LastEditTime: 2022-03-03 22:58:14
LastEditors: Please set LastEditors
Description: refer to "weighted speech distortion lossed for neural network based real-time speech enhancement"
FilePath: /CRUSE/test/test_loss.py
'''
import os
from matplotlib import pyplot as plt
import numpy as np
import librosa as lib
import soundfile as sf


def plot_mesh(img, title="", save_home=""):
    img = img
    fig, ax = plt.subplots()
    plt.title(title)
    fig.colorbar(plt.pcolormesh(range(img.shape[1]), range(img.shape[0]), img))
    if save_home != "":
        print(os.path.join(save_home, "%s.jpg" % title))
        plt.savefig(os.path.join(save_home, "%s.jpg" % title))
        return


def snr_weight_loss_adjust():
    global_snr = np.linspace(-10, 30, 410)
    beta = np.array([1, 5, 10, 20])
    # global_snr = np.expand_dims(global_snr, axis=0)
    # beta = np.expand_dims(beta, axis=-1)
    alpha = np.zeros((len(beta), len(global_snr)))
    for i in range(len(beta)):
        tmp = 10**(global_snr / 10)
        beta_tmp = 10**(beta[i] / 10)
        alpha[i] = tmp / (tmp + beta_tmp)
    plt.xlim(xmin=-10, xmax=30)
    plt.ylim(ymin=0., ymax=1.0)
    plt.plot(global_snr, alpha.T)
    plt.plot(global_snr, np.ones_like(global_snr) * 0.5, 'k:')
    plt.xlabel('global snr of the noisy utterence in training(dB)')
    plt.ylabel('speech distortion weight(alpha)')
    plt.legend(['b=1dB', 'b=5dB', 'b=10dB', 'b=20dB'])
    plt.show()
    print('sc')


def speech_distortion_test():
    main_path = '/Users/audio_source/GaGNet/First_DNS_no_reverb'
    file_name = 'clnsp1_train_69005_1_snr15_tl-21_fileid_158.wav'
    clean_name = os.path.join(main_path, 'no_reverb_clean',
                              'clean_fileid_158.wav')
    noise_name = os.path.join(main_path, 'no_reverb_mix', file_name)
    clean, _ = lib.load(clean_name, sr=16000)
    noisy, _ = lib.load(noise_name, sr=16000)
    noise = noisy - clean

    win_len = 512
    hop_len = 128
    n_fft = 512
    stft_clean = lib.stft(clean,
                          win_length=win_len,
                          hop_length=hop_len,
                          n_fft=n_fft)
    stft_noise = lib.stft(noise,
                          win_length=win_len,
                          hop_length=hop_len,
                          n_fft=n_fft)
    stft_noisy = lib.stft(noisy,
                          win_length=win_len,
                          hop_length=hop_len,
                          n_fft=n_fft)
    clean_mag, _ = lib.magphase(stft_clean)
    noise_mag, _ = lib.magphase(stft_noise)
    noisy_mag, noisy_phase = lib.magphase(stft_noisy)
    # plot_mesh(np.log(clean_mag), 'clean_mag')
    # plot_mesh(np.log(noise_mag), 'noise_mag')
    # plt.show()

    snr = clean_mag / (noise_mag + clean_mag)
    # snr = clean_mag / noise_mag
    # snr = clean_mag / noisy_mag
    enhance_noisy = noisy_mag * snr
    enhance_noise = noise_mag * snr
    enhance_clean = clean_mag * snr
    # plot_mesh(np.log(enhance_noisy), 'enhance_noisy')
    # plot_mesh(np.log(enhance_noise), 'enhance_noise')
    # plot_mesh(np.log(enhance_clean), 'enhance_clean')
    # plt.show()

    enhance_noisy_tmp = enhance_noisy * noisy_phase
    enhance_noise_tmp = enhance_noise * noisy_phase
    enhance_clean_tmp = enhance_clean * noisy_phase
    len_noisy = len(noisy)
    enhance_noisy_t = lib.istft(enhance_noisy_tmp,
                                win_length=win_len,
                                hop_length=hop_len,
                                length=len_noisy)
    enhance_clean_t = lib.istft(enhance_clean_tmp,
                                win_length=win_len,
                                hop_length=hop_len,
                                length=len_noisy)
    enhance_noise_t = lib.istft(enhance_noise_tmp,
                                win_length=win_len,
                                hop_length=hop_len,
                                length=len_noisy)
    sf.write("./enhance_noisy_t.wav", enhance_noisy_t, samplerate=16000)
    sf.write("./enhance_clean_t.wav", enhance_clean_t, samplerate=16000)
    sf.write('./enhance_noise_t.wav', enhance_noise_t, samplerate=16000)

    print('sc')


if __name__ == "__main__":
    # snr_weight_loss_adjust()
    speech_distortion_test()
