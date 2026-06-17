import os
import pandas
import glob
from tqdm import tqdm
import librosa
import librosa.display
import matplotlib.pyplot
import numpy as np
from sklearn.model_selection import train_test_split
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
        
    for Data_index in tqdm(sound_path_dictionaly,desc="preprocessing"):
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

    #PyTorchに入れる
    train_tensor = torch.tensor(train_data)
    val_normal_tensor  = torch.tensor(val_normal_data)
    val_abnormal_tensor  = torch.tensor(val_abnormal_data)

    train_loader=DataLoader(train_data, batch_size=1, shuffle=True)
    val_normal_loader=DataLoader(val_normal_tensor, batch_size=1, shuffle=True)
    val_abnormal_loader=DataLoader(val_abnormal_tensor, batch_size=1, shuffle=True)

    # ResNetの構造がわからない場合はResNetの構造を出力して確認しましょう。
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

    for epoch in tqdm(range(10)):
        # 学習
        train_loss = 0

        for x in train_loader:
            optimizer.zero_grad()

            x = x.to(device)
            x = x.unsqueeze(1) # [Batch, 1, 128, 474] の形にする

            out = resnet_model(x)
            out = out.view(x.size(0), 1, 128, 474) # 出力を入力と同じ形に変形

            loss = loss_function(out, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1} - Train Loss (正常の復元エラー): {train_loss:.4f}")

        # 検証
        print(f"異常音検知開始")
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
        print(f"終了")
        print(f"検証用 正常音の平均エラー値: {np.mean(normal_loss):.4f}")
        print(f"検証用 異常音の平均エラー値: {np.mean(abnormal_loss):.4f}")



#初回ログpreprocessing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:11<00:00, 88.49it/s]
#preprocessing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 350/350 [00:03<00:00, 89.66it/s]
#preprocessing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 264/264 [00:02<00:00, 88.57it/s]
#Epoch 1 - Train Loss (正常の復元エラー): 533069.8008
#異常音検知開始
#終了
#検証用 正常音の平均エラー値: 46.6869
#検証用 異常音の平均エラー値: 76.1982

#Epoch 2 - Train Loss (正常の復元エラー): 53265.7057
#異常音検知開始
#終了
#検証用 正常音の平均エラー値: 48.5111
#検証用 異常音の平均エラー値: 75.3041
#Epoch 3 - Train Loss (正常の復元エラー): 48525.9086
#異常音検知開始
#終了
#検証用 正常音の平均エラー値: 47.2244
#検証用 異常音の平均エラー値: 73.9516
#Epoch 4 - Train Loss (正常の復元エラー): 48304.5259
#異常音検知開始
#終了
#検証用 正常音の平均エラー値: 50.3384
#検証用 異常音の平均エラー値: 74.4547
#Epoch 5 - Train Loss (正常の復元エラー): 47894.3153
#異常音検知開始
#終了
#検証用 正常音の平均エラー値: 46.3967
#検証用 異常音の平均エラー値: 71.2743
#Epoch 6 - Train Loss (正常の復元エラー): 41058.7057
#異常音検知開始
#終了
#検証用 正常音の平均エラー値: 40.3647
#検証用 異常音の平均エラー値: 64.4088
#Epoch 7 - Train Loss (正常の復元エラー): 38279.2194
#異常音検知開始
#終了
#検証用 正常音の平均エラー値: 36.3428
#検証用 異常音の平均エラー値: 60.2631
#Epoch 8 - Train Loss (正常の復元エラー): 35956.4966
#異常音検知開始
#終了
#検証用 正常音の平均エラー値: 35.8817
#検証用 異常音の平均エラー値: 56.0945
#Epoch 9 - Train Loss (正常の復元エラー): 35506.2610
#異常音検知開始
#終了
#検証用 正常音の平均エラー値: 35.7616
#検証用 異常音の平均エラー値: 55.0205
#Epoch 10 - Train Loss (正常の復元エラー): 35169.5558
#異常音検知開始
#終了
#検証用 正常音の平均エラー値: 36.0154
#検証用 異常音の平均エラー値: 55.5176