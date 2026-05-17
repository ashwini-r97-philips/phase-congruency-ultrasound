"""Standard UNet encoder-decoder with skip connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims mismatch due to odd input sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, num_classes=1):
        """
        encoder_channels: [in_ch, e1, e2, e3, e4]  e.g. [1, 32, 64, 128, 256]
        decoder_channels: [d1, d2, d3, d4]          e.g. [256, 128, 64, 32]
        bottleneck uses 2 * encoder_channels[-1]
        """
        super().__init__()
        ec = encoder_channels
        dc = decoder_channels
        bottleneck_ch = ec[-1] * 2  # 512

        self.encoders = nn.ModuleList([
            EncoderBlock(ec[i], ec[i + 1]) for i in range(len(ec) - 1)
        ])
        self.bottleneck = ConvBlock(ec[-1], bottleneck_ch)
        self.decoders = nn.ModuleList([
            DecoderBlock(
                bottleneck_ch if i == 0 else dc[i - 1],
                ec[-(i + 2)],
                dc[i],
            )
            for i in range(len(dc))
        ])
        self.head = nn.Conv2d(dc[-1], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])

        return self.head(x)  # raw logits [B, 1, H, W]


def build_unet(cfg):
    mcfg = cfg["model"]
    return UNet(
        encoder_channels=mcfg["encoder_channels"],
        decoder_channels=mcfg["decoder_channels"],
        num_classes=mcfg["num_classes"],
    )


if __name__ == "__main__":
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = build_unet(cfg)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print("output shape:", out.shape)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n_params:,}")
