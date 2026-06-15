import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

ch_num = "1"

BASE_AUDIO_FILE_PATH = f"/run/media/wakincho/SSPQ-USC/Projects/Datasets/toy_admos_project/Training_SoundFiles/ToyCar/ch{ch_num}/train_normal/"
MUSIC_PATH = f"0{ch_num}0001_ToyCar_case1_normal_IND_ch{ch_num}_0001.wav"
# 音声データ（data）とサンプリングレート（fs）を読込
data, fs = sf.read(BASE_AUDIO_FILE_PATH + MUSIC_PATH)

time = np.arange(len(data)) / fs

plt.figure(figsize=(10, 4))

# 折れ線グラフを描画 (横軸: 時間[秒], 縦軸: 音の大きさ[振幅])
plt.plot(time, data, color="blue", linewidth=0.5)

# グラフ設定
plt.title("WAV Audio Waveform")
plt.xlabel("Time [seconds]")
plt.ylabel("Amplitude")

# 補助線を表示
plt.grid(True)

# グラフを保存
SAVE_PATH = f"/home/wakincho/Projects/DeepLearning_Projects/src_ToyAdmos/0.Create_Datasets_From_SoundFiles/Wave_png/0{ch_num}0001_ToyCar_case1_normal_IND_ch{ch_num}_0001"
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"グラフを画像として保存: {SAVE_PATH}")