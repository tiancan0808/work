import torch.nn.functional as F
from torch import nn
import torch


class SSIM(nn.Module):
    def __init__(self, win_size=9, k1=0.01, k2=0.03, gaussian_weights=True, use_sample_covariance=False, sigma=1.5,
                 in_channel=1):
        super().__init__()
        self.win_size = win_size
        self.pad_shp = [(win_size - 1) // 2] * 4
        self.k1, self.k2 = k1, k2
        self.in_channel = in_channel
        if gaussian_weights:
            assert win_size % 2 == 1
            coords = torch.arange(win_size, dtype=torch.float)
            coords -= win_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            _1D_window = g.unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
            if in_channel == 1:
                self.w = _2D_window.cuda()
            else:
                self.w = _2D_window.expand(in_channel, 1, win_size, win_size).cuda()
        else:
            self.w = torch.ones(in_channel, 1, win_size, win_size).div(win_size ** 2).cuda()
        if use_sample_covariance:
            NP = win_size ** 2
            self.cov_norm = NP / (NP - 1)
        else:
            self.cov_norm = 1

    def forward(self, X, Y, alpha=0.5, beta=1.0, data_range=2.0, use_pad=False, mode='reflect', return_full=False):
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        if use_pad:
            X_ = F.pad(X, self.pad_shp, mode=mode)
            Y_ = F.pad(Y, self.pad_shp, mode=mode)
            X_2 = F.pad(X * X, self.pad_shp, mode=mode)
            Y_2 = F.pad(Y * Y, self.pad_shp, mode=mode)
            XY = F.pad(X * Y, self.pad_shp, mode=mode)
        else:
            X_ = X
            Y_ = Y
            X_2 = X * X
            Y_2 = Y * Y
            XY = X * Y

        ux = F.conv2d(X_, self.w, groups=self.in_channel)
        uy = F.conv2d(Y_, self.w, groups=self.in_channel)

        uxx = F.conv2d(X_2, self.w, groups=self.in_channel)
        uyy = F.conv2d(Y_2, self.w, groups=self.in_channel)
        uxy = F.conv2d(XY, self.w, groups=self.in_channel)

        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)

        A1, A2, B1, B2 = 2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2
        D = B1 * B2
        S = (A1 * A2) / D

        if return_full:
            return S
        ssim_loss = (1 - S).mean()

        loss_l1 = F.l1_loss(X_, Y_, reduction='mean')
        mix = alpha * ssim_loss + beta * loss_l1

        return mix


if __name__ == '__main__':
    x = torch.randn((1, 3, 25, 25)).cuda()
    y = torch.randn((1, 3, 25, 25)).cuda()
    metric = SSIM(in_channel=3)
    res = metric(x, y, use_pad=True, return_full=True)
    print(res.mean(1).size())
