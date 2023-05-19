# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from math import ceil


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=in_channels // 16, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, factor=2.0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                              bias=False)
        self.norm = Normalize(out_channels)
        self.factor = factor

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode="nearest")
        x = self.conv(x)
        x = self.norm(x)
        x = nonlinearity(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, bias=False)
        self.norm = Normalize(out_channels)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), mode="reflect")
        x = self.conv(x)
        x = self.norm(x)
        x = nonlinearity(x)
        return x


class SEAttention(nn.Module):

    def __init__(self, channel, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.se = SEAttention(out_channels, 16)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               padding_mode='reflect',
                               bias=False)
        self.norm1 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               padding_mode='reflect',
                               bias=False)
        self.norm2 = Normalize(out_channels)

        self.shortcut1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, bias=False), Normalize(out_channels))

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = h + self.shortcut1(x)
        h = self.se(h)

        return h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=False)
        self.k = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=False)
        self.v = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=False)
        self.proj_out = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1).contiguous()  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1).contiguous()  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(self, *, ch, n_downsampling=3, dropout=0.0, resamp_with_conv=True, in_channels, use_attn=True,
                 resolution, z_channels, double_z=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.n_downsampling = n_downsampling
        self.resolution = resolution
        self.in_channels = in_channels
        self.attn = use_attn
        print(f'using attn: {bool(use_attn)}')
        self.down = nn.ModuleList()
        in_chs = [ch * 2 ** i for i in range(n_downsampling)]
        out_chs = [ch * 2 ** i for i in range(1, n_downsampling + 1)]

        self.in_block = nn.Conv2d(in_channels, ch, 3, 1, 1, padding_mode='reflect')

        for i_level in range(self.n_downsampling):
            block_in = in_chs[i_level]
            block_out = out_chs[i_level]
            down = nn.Module()
            down.block1 = ResnetBlock(block_in, block_out, dropout)
            down.downsample = Downsample(block_out, block_out)
            if i_level == self.n_downsampling - 1 and use_attn:
                down.attn = AttnBlock(block_out)
            down.block2 = ResnetBlock(block_out, block_out, dropout)
            self.down.append(down)

        # middle
        self.mid = nn.Sequential(AttnBlock(block_out) if use_attn else ResnetBlock(block_out, block_out, dropout),
                                 ResnetBlock(block_out, block_out, dropout))

        # end
        self.conv_out = nn.Conv2d(block_out, 2 * z_channels if double_z else z_channels, 3, 1, 1,
                                  padding_mode='reflect')

    def forward(self, x):

        # downsampling
        h = nonlinearity(self.in_block(x))

        for i_level in range(self.n_downsampling):
            h = self.down[i_level].block1(h)
            h = self.down[i_level].downsample(h)
            if i_level == self.n_downsampling - 1 and self.attn:
                h = self.down[i_level].attn(h)
            h = self.down[i_level].block2(h)

        # middle
        h = self.mid(h)

        # end
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, n_downsampling=3, dropout=0.0, resamp_with_conv=True, in_channels, use_attn=True,
                 resolution, z_channels, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.n_upsampling = n_downsampling
        self.resolution = resolution
        self.attn = use_attn
        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * 2 ** n_downsampling
        curr_res = resolution // 2 ** n_downsampling
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1, padding_mode='reflect')

        # middle
        self.mid = nn.Sequential(ResnetBlock(block_in, block_in, dropout),
                                 AttnBlock(block_in) if use_attn else ResnetBlock(block_in, block_in, dropout),
                                 ResnetBlock(block_in, block_in, dropout)
                                 )

        # upsampling
        self.up = nn.ModuleList()
        in_chs = [2 ** i for i in range(self.n_upsampling, 0, -1)]
        out_chs = [2 ** i for i in range(self.n_upsampling - 1, -1, -1)]
        for i_level in range(self.n_upsampling):
            block_in = ch * in_chs[i_level]
            block_out = ch * out_chs[i_level]
            up = nn.Module()
            up.block1 = ResnetBlock(block_in, block_out, dropout)
            if i_level == 0 and use_attn:
                up.attn = AttnBlock(block_out)
            up.upsample = Upsample(block_out, block_out)
            up.block2 = ResnetBlock(block_out, block_out, dropout)
            curr_res = curr_res * 2

            self.up.append(up)

            # end
        self.conv_out = nn.Conv2d(block_out, out_ch, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # z to block_in
        h = nonlinearity(self.conv_in(z))

        # middle
        h = self.mid(h)

        # upsampling
        for i_level in range(self.n_upsampling):
            h = self.up[i_level].block1(h)
            if i_level == 0 and self.attn:
                h = self.up[i_level].attn(h)
            h = self.up[i_level].upsample(h)
            h = self.up[i_level].block2(h)

        # end
        h = self.conv_out(h)
        return h




if __name__ == '__main__':
    from torchsummary import summary

    model = Decoder(ch=32, in_channels=1, out_ch=1, use_attn=False, resolution=224, z_channels=128)
    summary(model, (128, 28, 28), device='gpu')
