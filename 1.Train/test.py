import os
import glob
from tqdm import tqdm
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

ch_num = "1"

config = {
    "ch_num": ch_num,
    "training_sound_folder_dir" : f"/run/media/wakincho/SSPQ-USC/Projects/Datasets/toy_admos_project/Training_SoundFiles/ToyCar/ch{ch_num}/train_normal/",
    "normal_sound_for_validation": f"/run/media/wakincho/SSPQ-USC/Projects/Datasets/toy_admos_project/Training_SoundFiles/ToyCar/ch{ch_num}/test_normal/",
    "abnormal_sound_for_validation": f"/run/media/wakincho/SSPQ-USC/Projects/Datasets/toy_admos_project/Training_SoundFiles/ToyCar/ch{ch_num}/test_anomaly/"
}

def Prepare_train_test_data(data_source):
    # Sound_files配列へ,"training_sound_folder_dir"フォルダに存在するwavファイル名を追加して,ファイル名配列を作成
    sound_files = glob.glob(os.path.join(data_source, "*.wav"))

    # 音声データのパスいれる辞書変数を定義
    sound_path_dictionaly = []
    # 元メルスペクトログラム,dBメルスペクトログラムに変換した音声データ配列を保存する配列
    soundwaves = []

    for sound_path in sound_files:
        sound_path_dictionaly.append({"Sound Path": sound_path})# 音声データのパスを辞書変数にぶちこむ
        
    for Data_index in tqdm(sound_path_dictionaly,desc="preprocessing", ncols=100):
        target_file = Data_index["Sound Path"]

        waveform, sample_rate = librosa.load(target_file, sr=16000)#音声ファイルの波形とレートをロード
        #librosa.loadは22050Hzで読み込む

        soundwaves.append(waveform)
    return np.array(soundwaves)

class Audio1DAutoEncoder(nn.Module):
    def __init__(self):
        super(Audio1DAutoEncoder,self).__init__()
        
        self.encoder = nn.Sequential(
            
        )
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16,kernel_size=64, stride=4, padding=32)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,kernel_size=32,stride=2, padding=16)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(32,2)

    def forward(self,x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__=="__main__":
    sound = Prepare_train_test_data(config["training_sound_folder_dir"])
    print(sound.shape)#(1000,242550)
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

