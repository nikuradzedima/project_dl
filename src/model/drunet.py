import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class DRUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        channels=(32, 64, 128, 256),
        residual=True,
    ):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.residual = residual and in_channels == out_channels
        self.head = nn.Conv2d(in_channels, c0, kernel_size=3, padding=1)
        self.enc0 = ConvBlock(c0, c0)
        self.down1 = DownBlock(c0, c1)
        self.down2 = DownBlock(c1, c2)
        self.down3 = DownBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.up0 = UpBlock(c1, c0, c0)
        self.tail = nn.Conv2d(c0, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        e0 = self.enc0(self.head(x))
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        y = self.bottleneck(e3)
        y = self.up2(y, e2)
        y = self.up1(y, e1)
        y = self.up0(y, e0)
        y = self.tail(y)
        if self.residual:
            y = x + y
        return y
