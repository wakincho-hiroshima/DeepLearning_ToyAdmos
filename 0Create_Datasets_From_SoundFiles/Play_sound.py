import sounddevice as sd
import soundfile as sf

BASE_AUDIO_FILE_PATH = "/run/media/wakincho/SSPQ-USC/Projects/Datasets/toy_admos_project/Training_SoundFiles/ToyCar/ch1/train_normal/"
MUSIC_PATH = "010001_ToyCar_case1_normal_IND_ch1_0001.wav"
# 1. 音声データ（data）とサンプリングレート（fs）を読み込む
data, fs = sf.read(BASE_AUDIO_FILE_PATH + MUSIC_PATH)

# 2. 再生する
print("🔊 再生中...")
sd.play(data, fs)

# 3. 再生が終わるまでプログラムを待機させる
sd.wait()
print("🏁 終了")