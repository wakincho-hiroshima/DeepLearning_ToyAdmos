import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ch_num = "1"

BASE_AUDIO_FILE_PATH = f"/run/media/wakincho/SSPQ-USC/Projects/Datasets/toy_admos_project/Training_SoundFiles/ToyCar/ch{ch_num}/train_normal/"
MUSIC_PATH = f"0{ch_num}0001_ToyCar_case1_normal_IND_ch{ch_num}_0001.wav"

# 音声データ（data）とサンプリングレート（fs）を読込
data, fs = sf.read(BASE_AUDIO_FILE_PATH + MUSIC_PATH)

# ==========================================
# 1. 普通のFFT（高速フーリエ変換）の計算
# ==========================================
n = len(data)  # 全データ数（サンプル数）

# NumPyのFFTを実行（複素数が返ってくる）
fft_data = np.fft.fft(data)

# 複素数から「振幅（音の強さ）」を取り出す
amplitude = np.abs(fft_data)

# 【重要】実数データのFFTは左右対称になるため、左半分（正の周波数）だけを抽出
half_n = n // 2
amplitude_half = amplitude[:half_n]

# 横軸用の「周波数（Hz）」のリストを作る
frequencies = np.fft.fftfreq(n, d=1 / fs)[:half_n]

# ==========================================
# 2. グラフの描画と保存
# ==========================================
plt.figure(figsize=(10, 4))

# 折れ線グラフを描画 (横軸: 周波数[Hz], 縦軸: 振幅)
plt.plot(frequencies, amplitude_half, color="red", linewidth=0.8)

# グラフ設定
plt.title(f"FFT Spectrum (Full Audio) - CH{ch_num}")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")

# 横軸（周波数）の範囲を 0Hz 〜 最高周波数（サンプリングレートの半分：ナイキスト周波数）までに制限
plt.xlim(0, fs / 2)

# 補助線を表示
plt.grid(True)

# グラフを保存（ファイル名に「_FFT.png」を追加して区別できるようにしています）
SAVE_PATH = f"/home/wakincho/Projects/DeepLearning_Projects/src_ToyAdmos/0.Create_Datasets_From_SoundFiles/Wave_png/FFT_0{ch_num}0001_ToyCar_case1_normal_IND_ch{ch_num}_0001.png"

plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"📊 普通のFFTグラフを画像として保存: {SAVE_PATH}")