"""
NAFNet architecture — Nonlinear Activation Free Network (ECCV 2022)
Bundled directly to avoid dependency on megvii-research/NAFNet repo.
Source: https://github.com/megvii-research/NAFNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu  = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y   = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y   = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_tensors
        N, C, H, W = grad_output.size()
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g  = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = (g - y * mean_gy - mean_g) / (var + eps).sqrt()
        return (
            gx,
            (grad_output * y).sum(dim=(0, 2, 3)),
            grad_output.sum(dim=(0, 2, 3)),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias   = nn.Parameter(torch.zeros(channels))
        self.eps    = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_rate=0.0):
        super().__init__()
        dw_ch = c * dw_expand
        self.conv1   = nn.Conv2d(c, dw_ch, 1)
        self.conv2   = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.conv3   = nn.Conv2d(dw_ch // 2, c, 1)
        self.sca     = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        ffn_ch = ffn_expand * c
        self.conv4   = nn.Conv2d(c, ffn_ch, 1)
        self.conv5   = nn.Conv2d(ffn_ch // 2, c, 1)
        self.norm1   = LayerNorm2d(c)
        self.norm2   = LayerNorm2d(c)
        self.gate    = SimpleGate()
        self.drop    = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.beta    = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma   = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.drop(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.gate(x)
        x = self.conv5(x)
        x = self.drop(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16,
                 middle_blk_num=1,
                 enc_blk_nums=None,
                 dec_blk_nums=None):
        super().__init__()
        if enc_blk_nums is None: enc_blk_nums = []
        if dec_blk_nums is None: dec_blk_nums = []

        self.intro     = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending    = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders  = nn.ModuleList()
        self.decoders  = nn.ModuleList()
        self.middle_blks = nn.Sequential(*[NAFBlock(width * (2 ** len(enc_blk_nums)))
                                           for _ in range(middle_blk_num)])
        self.ups       = nn.ModuleList()
        self.downs     = nn.ModuleList()

        ch = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
            ch *= 2

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1),
                nn.PixelShuffle(2),
            ))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))

        self.padder = 16

    def check_image_size(self, x):
        _, _, h, w = x.size()
        ph = (self.padder - h % self.padder) % self.padder
        pw = (self.padder - w % self.padder) % self.padder
        return F.pad(x, (0, pw, 0, ph)), h, w

    def forward(self, inp):
        x, H, W = self.check_image_size(inp)
        x  = self.intro(x)
        enc_skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            enc_skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(enc_skips)):
            x = up(x)
            x = x + skip
            x = dec(x)
        x = self.ending(x)
        x = x + inp[:, :, :x.shape[2], :x.shape[3]]
        return x[:, :, :H, :W]