'''
Author: your name
Date: 2022-03-02 22:19:22
LastEditTime: 2022-03-02 22:51:51
LastEditors: Please set LastEditors
Description: refer to "weighted speech distortion lossed for neural network based real-time speech enhancement"
FilePath: /CRUSE/test/test_loss.py
'''
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    global_snr = np.linspace(-10, 30, 410)
    beta = np.array([1, 5, 10, 20])
    # global_snr = np.expand_dims(global_snr, axis=0)
    # beta = np.expand_dims(beta, axis=-1)
    alpha = np.zeros((len(beta), len(global_snr)))
    for i in range(len(beta)):
        tmp = 10**(global_snr / 10)
        beta_tmp = 10**(beta[i] / 10)
        alpha[i] = tmp / (tmp + beta_tmp)
    # speech_distortion = np.matmul(beta, global_snr).T
    plt.xlim(xmin=-10, xmax=30)
    plt.ylim(ymin=0., ymax=1.0)
    plt.plot(global_snr, alpha.T)
    plt.plot(global_snr, np.ones_like(global_snr) * 0.5, 'k:')
    plt.xlabel('global snr of the noisy utterence in training(dB)')
    plt.ylabel('speech distortion weight(alpha)')
    plt.legend(['b=1dB', 'b=5dB', 'b=10dB', 'b=20dB'])
    plt.show()
    print('sc')