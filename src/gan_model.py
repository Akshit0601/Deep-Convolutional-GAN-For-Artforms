import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def _weights_init(m):
    """DCGAN-style weight initialisation: Conv/ConvTranspose N(0, 0.02), BatchNorm N(1, 0.02)."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    DCGAN Generator.

    Architecture (for 64×64 RGB output, ngf=64):
        z(nz) → Dense → Reshape(ngf*8, 4, 4)
          → ConvTranspose2d(ngf*8 → ngf*4, 4,2,1) + BN + ReLU  → (ngf*4, 8, 8)
          → ConvTranspose2d(ngf*4 → ngf*2, 4,2,1) + BN + ReLU  → (ngf*2, 16, 16)
          → ConvTranspose2d(ngf*2 → ngf,   4,2,1) + BN + ReLU  → (ngf,   32, 32)
          → ConvTranspose2d(ngf   → nc,    4,2,1) + Tanh        → (nc,    64, 64)
    """

    def __init__(self, z_dim: int = 128, channels: int = 3, ngf: int = 64):
        super().__init__()
        self.z_dim = z_dim

        self.net = nn.Sequential(
            # Input: (z_dim) → (ngf*8, 4, 4)
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf*8, 4, 4) → (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4, 8, 8) → (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2, 16, 16) → (ngf, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf, 32, 32) → (channels, 64, 64)
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (B, z_dim) → reshape to (B, z_dim, 1, 1) for ConvTranspose2d
        return self.net(z.view(z.size(0), self.z_dim, 1, 1))


class Discriminator(nn.Module):
    """
    DCGAN Discriminator with Spectral Normalisation.

    Architecture (for 64×64 RGB input, ndf=64):
        (nc, 64, 64)
          → Conv2d(nc  → ndf,   4,2,1) + LeakyReLU(0.2)             → (ndf,   32, 32)
          → SN Conv2d(ndf  → ndf*2, 4,2,1) + BN + LeakyReLU(0.2)   → (ndf*2, 16, 16)
          → SN Conv2d(ndf*2→ ndf*4, 4,2,1) + BN + LeakyReLU(0.2)   → (ndf*4, 8, 8)
          → SN Conv2d(ndf*4→ ndf*8, 4,2,1) + BN + LeakyReLU(0.2)   → (ndf*8, 4, 4)
          → SN Conv2d(ndf*8→ 1,     4,1,0) + Sigmoid                → (1, 1, 1)
    """

    def __init__(self, channels: int = 3, ndf: int = 64, use_spectral_norm: bool = True):
        super().__init__()

        def _maybe_sn(layer):
            return spectral_norm(layer) if use_spectral_norm else layer

        self.net = nn.Sequential(
            # No BN on first layer (DCGAN guideline), no spectral norm here either
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf, 32, 32) → (ndf*2, 16, 16)
            _maybe_sn(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2, 16, 16) → (ndf*4, 8, 8)
            _maybe_sn(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4, 8, 8) → (ndf*8, 4, 4)
            _maybe_sn(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8, 4, 4) → (1, 1, 1)
            _maybe_sn(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)
