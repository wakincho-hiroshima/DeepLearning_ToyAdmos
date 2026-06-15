import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

ch_num = "1"

BASE_AUDIO_FILE_PATH = f"/run/media/wakincho/SSPQ-USC/Projects/Datasets/toy_admos_project/Training_SoundFiles/ToyCar/ch{ch_num}/train_normal/"
MUSIC_PATH = f"0{ch_num}0001_ToyCar_case1_normal_IND_ch{ch_num}_0001.wav"

SAVE_PATH = f"/home/wakincho/Projects/DeepLearning_Projects/src_ToyAdmos/0.Create_Datasets_From_SoundFiles/Wave_png/Spectrogram_0{ch_num}0001_ToyCar_case1_normal_IND_ch{ch_num}_0001.png"

data, sample_rate = librosa.load(BASE_AUDIO_FILE_PATH + MUSIC_PATH)

#fftで周波数解析
stft_data = librosa.stft(data)

spectrogram_dB = librosa.amplitude_to_db(np.abs(stft_data), ref=np.max)

plt.figure(figsize=(10,4))

librosa.display.specshow(spectrogram_dB, sr=sample_rate, x_axis='time', y_axis='log')

plt.title(f"Spectrogram_CH{ch_num}_0{ch_num}0001_ToyCar_case1_normal_IND_ch{ch_num}_0001.wav")
plt.colorbar(format="%+2.0f dB")  # 音の強さを表すカラーバーを右側に表示
plt.xlabel("Time [seconds]")
plt.ylabel("Frequency [Hz]")

plt.savefig(SAVE_PATH, dpi=300)
plt.close()

