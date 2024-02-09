import torch
from einops import rearrange

class ImageEncoder(torch.nn.Module):
    #image encoder
    def __init__(self, latent_dim=512):
        super(ImageEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = torch.nn.Conv2d(128, 512, 3, 2, 1)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((8, 8))
    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b c h w t -> (b t) c h w')
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x[:, :self.latent_dim]
        return x