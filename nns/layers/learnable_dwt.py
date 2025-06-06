import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_wavelet_filters(wavelet_name="haar"):
    wave = pywt.Wavelet(wavelet_name)
    h_dec = torch.tensor(wave.dec_lo, dtype=torch.float32)
    g_dec = torch.tensor(wave.dec_hi, dtype=torch.float32)
    return h_dec, g_dec


def symmetric_pad(x, pad_left, pad_right):
    if pad_left > 0:
        left = x[..., 0:pad_left].flip(dims=[-1])
    else:
        left = torch.tensor([], dtype=x.dtype, device=x.device).reshape(x.shape[0], x.shape[1], 0)

    if pad_right > 0:
        right = x[..., -pad_right:].flip(dims=[-1])
    else:
        right = torch.tensor([], dtype=x.dtype, device=x.device).reshape(x.shape[0], x.shape[1], 0)

    x_padded = torch.cat([left, x, right], dim=-1)
    return x_padded


def dwt_1d_single_level_sym(x, h, g):
    B, N, T = x.shape
    L = h.shape[0]

    h_filter = h.view(1, 1, -1).repeat(N, 1, 1)  # shape: (N,1,L)
    g_filter = g.view(1, 1, -1).repeat(N, 1, 1)

    right_pad = L-1 if (T+L) % 2 == 0 else L
    x_pad = symmetric_pad(x, L-1, right_pad)  # (B, N, T + 2*(L-1) or 2L-1)

    c_l = F.conv1d(x_pad, h_filter, stride=2, padding=0, groups=N)
    c_h = F.conv1d(x_pad, g_filter, stride=2, padding=0, groups=N)

    return c_l, c_h


def dwt_1d_multi_level(x, h, g, levels=1, mode='symmetric'):
    c_h_list = []
    current = x
    for k in range(levels):
        if mode == 'symmetric':
            c_l, c_h = dwt_1d_single_level_sym(current, h, g)
        else:
            raise Exception(f'Unknown mode {mode}')
        c_h_list.append(c_h)
        current = c_l  # feed low freq into next level
    return current, c_h_list  # current is C_l^K, c_h_list is [C_h^1, ..., C_h^K]


def idwt_1d_single_level_sym(c_l, c_h, h_tilde, g_tilde):
    B, N, T_down = c_l.shape
    L = h_tilde.shape[0]

    h_tilde_filter = h_tilde.view(1, 1, -1).repeat(N, 1, 1)
    g_tilde_filter = g_tilde.view(1, 1, -1).repeat(N, 1, 1)

    x_l_upsampled = F.conv_transpose1d(c_l, h_tilde_filter, stride=2, padding=L-1, groups=N)
    x_h_upsampled = F.conv_transpose1d(c_h, g_tilde_filter, stride=2, padding=L-1, groups=N)
    x_recon = x_l_upsampled + x_h_upsampled

    return x_recon


def idwt_1d_multi_level(c_lK, c_h_list, h_tilde, g_tilde, mode='symmetric'):
    current = c_lK
    # We invert from the last level back to the first
    for k in reversed(range(len(c_h_list))):
        c_l = current
        c_h = c_h_list[k]
        if mode == 'symmetric':
            if c_l.shape[-1] > c_h.shape[-1]:
                c_l = c_l[:, :, :-1]
            current = idwt_1d_single_level_sym(c_l, c_h, h_tilde, g_tilde)
        else:
            raise Exception(f"Unknown mode {mode}")
    return current


class AdaDecomp(nn.Module):
    def __init__(self, wavelet_name="haar", levels=1, filter_learn=True):
        super(AdaDecomp, self).__init__()
        self.levels = levels
        self.mode = 'symmetric'

        h_dec_init, g_dec_init = initialize_wavelet_filters(wavelet_name)

        self.h_dec = nn.Parameter(h_dec_init, requires_grad=filter_learn)
        self.g_dec = nn.Parameter(g_dec_init, requires_grad=filter_learn)

    def forward(self, X):
        """
        X: shape (B, T, N)
        Returns a dict of:
           'c_lK': final low-frequency coefficients, shape (B, N, T//(2^K))
           'c_h_list': list of high-freq coeff from level 1..K
           'x_l': the reconstructed low-frequency time series (B, T, N)
           'x_h_list': list of reconstructed high-frequency time series
                       each with shape (B, T, N)
        """
        X_in = X.permute(0, 2, 1)

        # 1) Multi-level decomposition
        c_lK, c_h_list = dwt_1d_multi_level(X_in, self.h_dec, self.g_dec,
                                            levels=self.levels, mode=self.mode)

        # 2) Reconstruct low-frequency time series X_l
        zero_h_list = [torch.zeros_like(c_h) for c_h in c_h_list]
        x_l = idwt_1d_multi_level(c_lK, zero_h_list, self.h_dec, self.g_dec, mode=self.mode)  # (B, N, T)

        # 3) Reconstruct each high-frequency band individually
        x_h_list = []
        zero_l = torch.zeros_like(c_lK)
        for k in range(self.levels):
            # Make a copy of zero_h_list
            tmp_h_list = [torch.zeros_like(c_h) for c_h in c_h_list]
            tmp_h_list[k] = c_h_list[k]
            x_h_k = idwt_1d_multi_level(zero_l, tmp_h_list, self.h_dec, self.g_dec, mode=self.mode)
            x_h_list.append(x_h_k)  # shape (B, N, T)

        # Permute outputs back to (B, T, N)
        x_l = x_l.permute(0, 2, 1)  # (B, T, N)
        x_h_list = [xh.permute(0, 2, 1) for xh in x_h_list[::-1]]  # xhK, ..., xh1
        return x_l, x_h_list
