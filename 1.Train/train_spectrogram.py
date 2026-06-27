import os
import pandas
import glob
from tqdm import tqdm
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.models import resnet34

ch_num = "1"

config = {
    "ch_num": ch_num,
    "training_sound_folder_dir" : f"/run/media/wakincho/SSPQ-USC/Projects/Datasets/toy_admos_project/Training_SoundFiles/ToyCar/ch{ch_num}/train_normal/",
    "normal_sound_for_validation": f"/run/media/wakincho/SSPQ-USC/Projects/Datasets/toy_admos_project/Training_SoundFiles/ToyCar/ch{ch_num}/test_normal/",
    "abnormal_sound_for_validation": f"/run/media/wakincho/SSPQ-USC/Projects/Datasets/toy_admos_project/Training_SoundFiles/ToyCar/ch{ch_num}/test_anomaly/"
}

def Prepare_train_test_data(data_type):
    # Sound_files配列へ,"training_sound_folder_dir"フォルダに存在するwavファイル名を追加して,ファイル名配列を作成
    sound_files = glob.glob(os.path.join(data_type, "*.wav"))

    # 音声データのパスいれる辞書変数を定義
    sound_path_dictionaly = []
    # 元メルスペクトログラム,dBメルスペクトログラムに変換した音声データ配列を保存する配列
    sound_features = []
    sound_features_dB = []

    for sound_path in sound_files:
        sound_path_dictionaly.append({"Sound Path": sound_path})# 音声データのパスを辞書変数にぶちこむ
        
    for Data_index in tqdm(sound_path_dictionaly,desc="preprocessing", ncols=100):
        target_file = Data_index["Sound Path"]

        waveform, sample_rate = librosa.load(target_file)#音声ファイルの波形とレートをロード

        feature_melspectrogram = librosa.feature.melspectrogram(y = waveform, sr = sample_rate)
        feature_melspectrogram_dB = librosa.amplitude_to_db(feature_melspectrogram, ref = np.max)

        sound_features.append(feature_melspectrogram)
        sound_features_dB.append(feature_melspectrogram_dB)

    return np.array(sound_features), np.array(sound_features_dB)
    
if __name__=="__main__":
    # 学習済みのResNetをダウンロード
    resnet_model = resnet34(weights=True)

    #GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #create data
    sound_features_for_train, sound_features_for_train_dB = Prepare_train_test_data(config["training_sound_folder_dir"])
    normal_sound_features_for_val, normal_sound_features_for_val_dB = Prepare_train_test_data(config["normal_sound_for_validation"])
    abnormal_sound_features_for_val, abnormal_sound_features_for_val_dB = Prepare_train_test_data(config["abnormal_sound_for_validation"])

    # 学習データと検証用データに分割
    # 分割必要ない。テストは
    # 元メルスペクトログラム,dBメルスペクトログラムでもどっちでもいい。好きなのえらんで
    train_data = sound_features_for_train_dB
    val_normal_data = normal_sound_features_for_val_dB
    val_abnormal_data = abnormal_sound_features_for_val_dB

    print(train_data.shape)

    #PyTorchに入れる
    train_tensor = torch.tensor(train_data)
    val_normal_tensor  = torch.tensor(val_normal_data)
    val_abnormal_tensor  = torch.tensor(val_abnormal_data)

    train_loader=DataLoader(train_tensor, batch_size=1, shuffle=True)
    val_normal_loader=DataLoader(val_normal_tensor, batch_size=1, shuffle=True)
    val_abnormal_loader=DataLoader(val_abnormal_tensor, batch_size=1, shuffle=True)

    # ResNetの構造がわからないときはResNetの構造を出力して確認しとく
    # print(resnet_model)
    # 最初の畳み込みのチャネル3をチャネル1に変更する
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # 最後の層の次元を今回のカテゴリ数に変更する
    resnet_model.fc = nn.Linear(512,128*474)

    # GPUつかう
    resnet_model = resnet_model.to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=2e-4)
    
    losses = []

    epoch_num = 50

    for epoch in tqdm(range(epoch_num), ncols=100):
        # 学習
        train_loss = 0

        print(f"\n--train--")
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoch_num}", ncols=100)
        for x in train_bar:
            optimizer.zero_grad()

            x = x.to(device)
            x = x.unsqueeze(1) # [Batch, 1, 128, 474] の形にする

            out = resnet_model(x)
            out = out.view(x.size(0), 1, 128, 474) # 出力を入力と同じ形に変形

            loss = loss_function(out, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}")

        # 検証
        print(f"--evaluation--")
        resnet_model.eval()

        normal_loss=[]
        abnormal_loss=[]

      # 正常音評価
        with torch.no_grad():
            for x in val_normal_loader:
                x = x.to(device).unsqueeze(1)
                out = resnet_model(x).view(x.size(0), 1, 128, 474)
                loss = loss_function(out, x)
                normal_loss.append(loss.item())

        # 異常音評価
        with torch.no_grad():
            for x in val_abnormal_loader:
                x = x.to(device).unsqueeze(1)
                out = resnet_model(x).view(x.size(0), 1, 128, 474)
                loss = loss_function(out, x)
                abnormal_loss.append(loss.item())

        # 結果の表示
        print(f"--completed--")
        print(f"normal sound loss: {np.mean(normal_loss):.4f}")
        print(f"abnomaly sound loss: {np.mean(abnormal_loss):.4f}")