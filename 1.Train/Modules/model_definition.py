import torch
import torch.nn as nn
import torch.nn.functional as F


def frame_concat(h, bw, fw, device):
    """
    (Batch, Feature, Time) の入力に対し、時間軸(dim=2)で前後フレームを結合する
    """
    batch, feat, time = h.shape
    # 結合用に元の h を保持
    combined = [h]

    # 後方（過去）フレーム: 過去の情報をずらして追加
    for ii in range(1, bw + 1):
        pad = torch.zeros((batch, feat, ii), device=device)
        combined.append(torch.cat([h[:, :, ii:], pad], dim=2))

    # 前方（未来）フレーム: 未来の情報をずらして追加
    for ii in range(1, fw + 1):
        pad = torch.zeros((batch, feat, ii), device=device)
        combined.append(torch.cat([pad, h[:, :, :-ii]], dim=2))

    return torch.cat(combined, dim=1)  # Feature次元(dim=1)に結合


class FCN_AE(nn.Module):
    def __init__(self, in_dim, hid_dim, z_dim, num_hid, num_fw, num_bw):
        super(FCN_AE, self).__init__()
        self.num_fw, self.num_bw = num_fw, num_bw
        concat_in_dim = in_dim * (1 + num_fw + num_bw)

        # Batch Normalizationは Feature軸(dim=1) に対して適用
        self.in_BN = nn.BatchNorm1d(in_dim, affine=False)

        self.e_in = nn.Linear(concat_in_dim, hid_dim)
        self.e_hidden = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(num_hid)])
        self.e_out = nn.Linear(hid_dim, z_dim)

        self.d_in = nn.Linear(z_dim, hid_dim)
        self.d_hidden = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(num_hid)])
        self.d_out = nn.Linear(hid_dim, concat_in_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def input_bn(self, x):
        # x: (Batch, Feature, Time) -> BatchNorm1dは (Batch, Feature, Time) も受け取れる
        return self.in_BN(x)

    def forward(self, x_data):
        device = x_data.device
        x = self.input_bn(x_data)
        x = frame_concat(x, self.num_bw, self.num_fw, device)

        # Linear層は最後の次元(Time)に適用されるため、permuteで調整
        h = x.permute(0, 2, 1)  # (B, Time, Concat_Feature)

        # Encoder
        h = F.relu(self.e_in(h))
        for layer in self.e_hidden:
            h = F.relu(layer(h))
        z = F.relu(self.e_out(h))

        # Decoder
        h = F.relu(self.d_in(z))
        for layer in self.d_hidden:
            h = F.relu(layer(h))
        y = self.d_out(h)

        # 誤差計算
        score = torch.mean((x.permute(0, 2, 1) - y) ** 2, dim=2)
        return score, x.permute(0, 2, 1), y
