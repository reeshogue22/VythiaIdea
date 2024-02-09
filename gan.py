import torch
import torch.nn as nn
from einops import rearrange


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.style_dim = style_dim
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_scale_transform = nn.Linear(style_dim, num_features)
        self.style_shift_transform = nn.Linear(style_dim, num_features)
    def forward(self, x, style):
        style = rearrange(style, 'b t c -> b (t c)')
        x = self.norm(x)
        style_scale = self.style_scale_transform(style)[:, :, None, None]

        style_shift = self.style_shift_transform(style)[:, :, None, None]
        return style_scale * x + style_shift

class UNetBlock(nn.Module):
    #AdaIN -> GELU -> Conv2d -> AdaIN -> GELU -> Conv2d -> Residual
    def __init__(self, in_channels, out_channels, style_dim):
        super(UNetBlock, self).__init__()
        self.norm1 = AdaptiveInstanceNorm2d(style_dim, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.gelu = nn.GELU()
    def forward(self, x, style):    
        res = x
        x = self.norm1(x, style)
        x = self.gelu(x)
        x = self.conv1(x)
        return x + res


class UpBlock(nn.Module):
    #Upsample -> Conv2d
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.gelu = nn.GELU()
    def forward(self, x):
        x = self.gelu(x)
        x = self.upsample(x)
        x = self.conv(x)
        return x
    
class DownBlock(nn.Module):
    #Conv2d with stride 2
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.gelu = nn.GELU()
    def forward(self, x):
        x = self.gelu(x)
        x = self.conv(x)
        return x

class StyleMapping(nn.Module):
    def __init__(self, style_dim, n_mlp, latent_dim):
        super(StyleMapping, self).__init__()
        self.style_dim = style_dim
        self.n_mlp = n_mlp
        self.latent_dim = latent_dim
        self.inp  = nn.Embedding(latent_dim, style_dim)
        self.mlp = nn.Sequential(*[nn.GELU(), nn.Linear(style_dim, style_dim)]*n_mlp)
    def forward(self, latent):
        style = self.inp(latent)
        return self.mlp(style)

class Generator(nn.Module):
    #UNet style generator with adaptive instance normalization
    def __init__(self, scaling=16, dim=3, style_dim=256, latent_dim=512, n_blocks=1):
        super(Generator, self).__init__()
        self.scaling = scaling
        self.dim = dim
        self.style_dim = style_dim
        self.latent_dim = latent_dim
        self.n_blocks = n_blocks

        self.to_scaling = nn.Conv2d(dim, scaling, 3, 1, 1)
        self.down1 = DownBlock(scaling, scaling*2)
        self.block2 = UNetBlock(scaling*2, scaling*2, style_dim)
        self.down2 = DownBlock(scaling*2, scaling*4)
        self.block3 = UNetBlock(scaling*4, scaling*4, style_dim)
        self.down3 = DownBlock(scaling*4, scaling*8)
        self.block4 = UNetBlock(scaling*8, scaling*8, style_dim)
        self.middle = UNetBlock(scaling*8, scaling*8, style_dim)
        self.up1 = UpBlock(scaling*8, scaling*8)
        self.block5 = UNetBlock(scaling*8, scaling*8, style_dim)
        self.up2 = UpBlock(scaling*8, scaling*4)
        self.block6 = UNetBlock(scaling*4, scaling*4, style_dim)
        self.up3 = UpBlock(scaling*4, scaling*2)
        self.block7 = UNetBlock(scaling*2, scaling*2, style_dim)
        self.up4 = UpBlock(scaling*2, scaling)
        self.out = nn.Conv2d(scaling, dim, 3, 1, 1)

        self.style_map = StyleMapping(style_dim, 4, latent_dim)
    def forward(self, x, latent):
        time = x.shape[-1]
        x = rearrange(x, 'b c h w t -> (b t) c h w')
        style = self.style_map(latent)
        x = self.to_scaling(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.block3(x, style)
        x = self.down3(x)
        x = self.block4(x, style)
        x = self.middle(x, style)
        x = self.up1(x)
        x = self.block5(x, style)
        x = self.up2(x)
        x = self.block6(x, style)
        x = self.up3(x)
        x = self.up4(x)
        x = self.out(x)
        x = rearrange(x, '(b t) c h w -> b c h w t', t=time)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, 1, 1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(8, 32, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(128, 1024)
    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b c h w t -> (b t) c h w')
        x = self.pool(torch.nn.functional.relu(self.conv(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x