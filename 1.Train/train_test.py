import os

import Modules.Config as Config
import Modules.model_definition as model_definition
import Modules.my_modules as my_modules
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sp_param, dnn_param, _ = Config.load_config()
toy_type = "ToyCar"

# モデルのロード
dnn_model = model_definition.FCN_AE(
    dnn_param["NumFB"],
    dnn_param["hid_dim"],
    dnn_param["z_dim"],
    dnn_param["num_hid"],
    dnn_param["num_fw"],
    dnn_param["num_bw"],
).to(device)
dnn_model.load_state_dict(torch.load(f"/home/wakincho/Projects/DeepLearning_Projects/src_ToyAdmos/1.Train/{toy_type}.pth"))
dnn_model.eval()

# テストデータの読み込み
test_dir = f"/home/wakincho/Projects/DeepLearning_Projects/src_ToyAdmos/exp1_dataset_{toy_type}/test_anomaly"
S_all, fn_all = my_modules.wav_read_test(test_dir, sp_param["wav_read_gain"])

print("Starting Evaluation...")
for i, x in enumerate(S_all):
    # 1件ずつ評価
    fet = my_modules.process_to_tensor([x], sp_param, device)
    with torch.no_grad():
        _, x_out, _ = dnn_model(fet)
        # 再構成誤差を異常スコアとする
        anomaly_score = torch.mean(torch.abs(fet - x_out))
    print(f"File: {fn_all[i]}, Anomaly Score: {anomaly_score.item():.4f}")