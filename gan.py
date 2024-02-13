import torch
import torch.nn as nn
from einops import rearrange

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_scale_transform = nn.Linear(style_dim, num_features)
        self.style_shift_transform = nn.Linear(style_dim, num_features)

    def forward(self, x, style):
        style = rearrange(style, 'b t c -> b (t c)')
        x = self.norm(x)
        style_scale = self.style_scale_transform(style)[:, :, None, None]
        style_shift = self.style_shift_transform(style)[:, :, None, None]
        return style_scale * x + style_shift

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super(ConvBlock, self).__init__()
        self.norm = AdaptiveInstanceNorm2d(style_dim, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.gelu = nn.GELU()

    def forward(self, x, style):
        res = x
        x = self.norm(x, style)
        x = self.gelu(x)
        x = self.conv(x)
        return x + res

class StyleMapping(nn.Module):
    def __init__(self, style_dim, n_mlp, latent_dim):
        super(StyleMapping, self).__init__()
        self.style_dim = style_dim
        self.n_mlp = n_mlp
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(*[nn.GELU(), nn.Linear(style_dim, style_dim)]*n_mlp)

    def forward(self, latent):
        style = nn.functional.embedding(latent, nn.Embedding(self.latent_dim, self.style_dim).weight)
        return self.mlp(style)

class Generator(nn.Module):
    def __init__(self, scaling=16, dim=3, style_dim=256, latent_dim=512, n_blocks=1):
        super(Generator, self).__init__()
        self.scaling = scaling
        self.style_dim = style_dim
        self.latent_dim = latent_dim
        self.n_blocks = n_blocks

        self.to_scaling = nn.Conv2d(dim, scaling, 3, 1, 1)
        self.blocks = nn.ModuleList([
            ConvBlock(scaling * (2 ** i), scaling * (2 ** (i + 1)), style_dim) for i in range(n_blocks)
        ])
        self.middle = ConvBlock(scaling * (2 ** n_blocks), scaling * (2 ** n_blocks), style_dim)
        self.upsamples = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBlock(
                scaling * (2 ** (i + 1)), scaling * (2 ** i), style_dim)) for i in range(n_blocks - 1, -1, -1)
        ])
        self.out = nn.Conv2d(scaling, dim, 3, 1, 1)
        self.style_map = StyleMapping(style_dim, 4, latent_dim)

    def forward(self, x, latent):
        time = x.shape[-1]
        x = rearrange(x, 'b c h w t -> (b t) c h w')
        style = self.style_map(latent)
        x = self.to_scaling(x)

        for block in self.blocks:
            x = block(x, style)

        x = self.middle(x, style)

        for upsample in self.upsamples:
            x = upsample(x)

        x = self.out(x)
        x = rearrange(x, '(b t) c h w -> b c h w t', t=time)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 32, 3, 1, 1),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, 1024)

    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b c h w t -> (b t) c h w')
        x = self.blocks(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
