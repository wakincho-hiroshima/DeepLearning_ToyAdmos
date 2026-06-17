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

ch_num = 1

config = {
    "ch_num": ch_num,
    "Sound_folder_dir" : f"/run/media/wakincho/SSPQ-USC/Projects/Datasets/toy_admos_project/Training_SoundFiles/ToyCar/{ch_num}/train_normal/"
}

# Sound_files配列へ,"Sound_folder_dir"フォルダに存在するwavファイル名を追加して,ファイル名配列を作成
Sound_files = glob.glob(os.path.join(config["Sound_folder_dir"], ".wav"))

def prepare_data():
    # 音声データを辞書変数にぶちこむ
    data_dictionaly = []

    for sound_path in Sound_files:
        data_dictionaly.append({"sound": sound_path})

    print(f"{len(data_dictionaly)} sounds found.")

# 画像をメルスペクトログラムに変換
class DatsUnits(Dataset):
    def __init__(self, base_path, df, in_col, out_col):
        self.df = df
        self.data = [] # 音源データをメルスペクトログラム(画像)に変換して格納する用
        self.labels =[] # 各データのカテゴリ情報を格納
        self.category_to_id = {}
        self.id_to_catefory = {}
        self.categories = list(sorted(df[out_col].unique())) # 正解ラベル格納用

        for i,category in enumerate(self.categories):
            self.category_to_id[category] = i
            self.id_to_catefory[i] = category

        for row in tqdm(range(len(df))):
            row = df.iloc[row]
            file_path = os.path.join(base_path, row[in_col])
            waveform, sample_rate = librosa.load(file_path)

            feature_melspec = librosa.feature.melspectrogram(y = waveform, sr=sample_rate)
            feature_melspec_db = librosa.power_to_db(feature_melspec, ref=np.max)

            self.data.append(feature_melspec_db)
            self.labels.append(self.category_to_id[row['catefory']])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
# 学習データと検証用データに分割
train_df, test_df = train_test_split(meta_df, train_size=0.8)