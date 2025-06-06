import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nns.layers.learnable_dwt import AdaDecomp


def diff(x):
    """
    First order differencing
    :param x: (*, T)
    :return:
    """
    return x[..., 1:] - x[..., :-1]


class Model(nn.Module):
    def __init__(
            self,
            seq_len,
            pred_len,
            wavelet="haar",
            filter_learn=True,
            dwt_levels=1,
            window_len=5,
            d_model=512,
            d_ff=1024,
            dropout=0.1,
            ffn_layers=1
    ):
        super().__init__()
        self.ada_decomp = AdaDecomp(wavelet, dwt_levels, filter_learn=filter_learn)
        self.window_len = window_len
        self.pad = nn.ReplicationPad1d(padding=(window_len // 2, window_len // 2 - ((window_len + 1) % 2)))
        self.epsilon = 1e-5
        self.mlp = StaticsMLP(seq_len, pred_len, d_model, d_ff, dropout=dropout, layer=ffn_layers)

    def sliding_std(self, x):
        # x: (..., T)
        x_window = x.unfold(-1, self.window_len, 1)
        s = x_window.std(dim=-1)
        s = self.pad(s)
        x = x / (s + self.epsilon)
        return x, s

    def normalize(self, x, predict=True):
        # x: (B, T, N)
        xl, xh_list = self.ada_decomp(x)  # (B, T, N)
        xh = torch.stack(xh_list, dim=-1).sum(dim=-1).transpose(-1, -2)  # (B, N, T)
        xh_norm, xh_s = self.sliding_std(xh)  # (B, N, T)

        if not predict:
            return xh_norm.transpose(-1, -2), xl, xh_s.transpose(-1, -2)  # (B, T, N)

        # stats prediction
        xl = xl.transpose(-1, -2)  # (B, N, T)
        mean_pred, std_pred = self.mlp(xh_s, xl, xh)  # (B, N, T)
        stats_pred = torch.stack([mean_pred, std_pred], dim=1).transpose(-1, -2)  # (B, 2, T, N)

        return xh_norm.transpose(-1, -2), stats_pred

    def de_normalize(self, x, stats_pred):
        mean = stats_pred[:, 0, ...]
        std = stats_pred[:, 1, ...]
        return x * (std + self.epsilon) + mean


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, activation, drop_rate=0.1, bias=False):
        super(FFN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias), activation,
            nn.Linear(d_ff, d_model, bias=bias), nn.Dropout(drop_rate),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class StaticsMLP(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, d_ff, dropout=0.1, layer=1, bias=False):
        super().__init__()
        # input embedding
        self.xl_proj = nn.Sequential(nn.Linear(seq_len, d_model), nn.Dropout(dropout))
        self.xl_diff_proj = nn.Sequential(nn.Linear(seq_len - 1, d_model), nn.Dropout(dropout))
        self.xh_proj = nn.Sequential(nn.Linear(seq_len, d_model), nn.Dropout(dropout))
        self.std_proj = nn.Sequential(nn.Linear(seq_len, d_model), nn.Dropout(dropout))

        # concat
        self.m_concat = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.Dropout(dropout))
        self.s_concat = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.Dropout(dropout))

        # ffn
        self.mean_ffn = nn.Sequential(*[FFN(d_model, d_ff, nn.GELU(), dropout, bias) for _ in range(layer)])
        self.std_ffn = nn.Sequential(*[FFN(d_model, d_ff, nn.GELU(), dropout, bias) for _ in range(layer)])

        # pred
        self.mean_pred = nn.Linear(d_model, pred_len)
        self.std_pred = nn.Linear(d_model, pred_len)

    def forward(self, std, xl, xh):
        # inputs: (B, N, T)
        m_all, s_all = xl.mean(dim=-1, keepdim=True), std.mean(dim=-1, keepdim=True)

        # input embedding
        xl, std = xl - m_all, std - s_all
        xl_diff = diff(xl)
        m_embed, s_embed = self.xl_proj(xl), self.std_proj(std)  # (B, N, d_model)
        xl_diff_embed = self.xl_diff_proj(xl_diff)
        xh_embed = self.xh_proj(xh)

        # process and concat
        m_concat = self.m_concat(torch.cat([m_embed, xl_diff_embed, xh_embed], dim=-1))
        s_concat = self.s_concat(torch.cat([s_embed, xh_embed, m_embed], dim=-1))

        # ffn and pred
        m_ffn, s_ffn = self.mean_ffn(m_concat), self.std_ffn(s_concat)
        mean, std = self.mean_pred(m_ffn) + m_all, self.std_pred(s_ffn) + s_all
        return mean, F.relu(std)





