import torch
from einops import rearrange

class ImageEncoder(torch.nn.Module):
    def __init__(self, latent_dim=512):
        super(ImageEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 2, 1),
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.Conv2d(64, 128, 3, 2, 1),
            torch.nn.Conv2d(128, 512, 3, 2, 1),
            torch.nn.AdaptiveAvgPool2d((8, 8))
        )

    def forward(self, x):
        b, c, h, w, t = x.shape
        x = rearrange(x, 'b c h w t -> (b t) c h w')
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = x[:, :self.latent_dim]
        return x
