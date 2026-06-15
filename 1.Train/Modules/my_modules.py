import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from tqdm import tqdm


def wavread(fn):
    fs, data = wavfile.read(fn)
    return data.astype(np.float32) / 32768.0, fs


def wav_read_trn(obs_dir, wav_per_set, wav_read_gain):
    obs_files = glob.glob(os.path.join(obs_dir, "*.wav"))
    obs_trn_all = []
    X_set = []
    for fn in tqdm(obs_files, desc="Loading Train"):
        x, _ = wavread(fn)
        X_set.append(x * wav_read_gain)
        if len(X_set) == wav_per_set:
            obs_trn_all.append(np.array(X_set))
            X_set = []
    return obs_trn_all


def wav_read_test(wav_dir, wav_read_gain):
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    S_all, fn_all = [], []
    for fn in tqdm(wav_files, desc="Loading Test"):
        x, _ = wavread(fn)
        S_all.append(x * wav_read_gain)
        fn_all.append(os.path.basename(fn))
    return S_all, fn_all


def process_to_tensor(x_list, sp_param, device):
    # すべてのデータを結合して1つのテンソルに
    x = torch.tensor(np.concatenate(x_list), dtype=torch.float32).to(device)

    # STFT実行
    window = torch.hann_window(sp_param["fftl"]).to(device)
    stft = torch.stft(
        x, n_fft=sp_param["fftl"], hop_length=sp_param["shift"], window=window, center=False, return_complex=True
    )

    # パワースペクトル -> 対数メルスペクトログラム
    power = torch.abs(stft) ** 2
    log_mel = torch.log10(power + 1e-6)

    # 次元を (1, Freq, Time) にする
    log_mel = log_mel.unsqueeze(0)

    return log_mel
