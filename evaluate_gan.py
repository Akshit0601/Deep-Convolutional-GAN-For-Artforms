#!/usr/bin/env python3
"""
Evaluation Script for trained DCGAN.
Loads a saved generator checkpoint, synthesizes images from latent prior N(0, I),
computes statistics, and generates image grids saved to outputs/.
"""

import os
from typing import Tuple, Dict, Any
import torch
from torchvision.utils import save_image, make_grid

from gan_config import default_gan_config, GANConfig
from src.gan_model import Generator, Discriminator


# ==============================================================================
# Model Checkpoint & Evaluation Parameters
# ==============================================================================
CKPT_PATH = "checkpoints/dcgan/best_dcgan.pt"
NUM_SAMPLES = 64
OUTPUT_DIR = "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==============================================================================


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Converts [-1, 1] normalized images to [0, 1]."""
    return torch.clamp(tensor * 0.5 + 0.5, 0.0, 1.0)


def load_gan_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[Generator, Discriminator, Dict[str, Any]]:
    if not os.path.exists(ckpt_path):
        latest_path = "checkpoints/dcgan/latest_dcgan.pt"
        if os.path.exists(latest_path):
            print(f"[Warning] '{ckpt_path}' not found, falling back to '{latest_path}'")
            ckpt_path = latest_path

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading DCGAN checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: GANConfig = checkpoint.get("config", default_gan_config)

    generator = Generator(z_dim=cfg.z_dim, channels=cfg.channels, ngf=cfg.ngf).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    discriminator = Discriminator(channels=cfg.channels, ndf=cfg.ndf, use_spectral_norm=cfg.spectral_norm).to(device)
    if "discriminator_state_dict" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        discriminator.eval()

    epoch = checkpoint.get("epoch", "N/A")
    metrics = checkpoint.get("metrics", {})
    print(f"Checkpoint loaded successfully (Epoch: {epoch}, Metrics: {metrics})")

    return generator, discriminator, checkpoint


def main():
    print("=" * 70)
    print("               DCGAN Generation & Evaluation")
    print("=" * 70)

    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # 1. Load Checkpoint
    generator, discriminator, checkpoint_info = load_gan_checkpoint(CKPT_PATH, device)
    cfg: GANConfig = checkpoint_info.get("config", default_gan_config)

    # 2. Sample from N(0, I)
    print(f"\nSynthesizing {NUM_SAMPLES} samples from latent prior N(0, I)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(NUM_SAMPLES, cfg.z_dim, device=device)
        fake_images = generator(z)  # (N, C, H, W) directly from conv generator
        fake_preds = discriminator(fake_images)
        fake_disp = denormalize(fake_images)

        # Create 8x8 grid for 64 samples
        grid = make_grid(fake_disp, nrow=8, normalize=False)
        output_grid_path = os.path.join(OUTPUT_DIR, "dcgan_generated_samples.png")
        save_image(grid, output_grid_path)

    avg_d_score = fake_preds.mean().item()
    print(f"\nSaved synthetic sample grid to: {output_grid_path}")
    print("-" * 70)
    print(f"Discriminator avg confidence on generated fakes D(G(z)): {avg_d_score:.4f} (Ideal ~0.50)")
    print("=" * 70)


if __name__ == "__main__":
    main()
