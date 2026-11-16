#!/usr/bin/env python3
"""
Generate N random synthetic samples from the best trained GAN/DCGAN checkpoint,
and extract the Top-K most realistic images according to the Discriminator score D(G(z)).

Usage:
    python sample_gan.py                  # Generates 50 samples and saves top 6
    python sample_gan.py --n 100 --top-k 6
    python sample_gan.py --n 50 --top-k 10 --output-dir outputs/my_samples
"""

import os
import argparse
import torch
from torchvision.utils import make_grid, save_image

from gan_config import default_gan_config, GANConfig
from src.gan_model import Generator, Discriminator


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Converts [-1, 1] normalized images to [0, 1]."""
    return torch.clamp(tensor * 0.5 + 0.5, 0.0, 1.0)


def generate_samples(
    n: int = 50,
    top_k: int = 6,
    ckpt_path: str = "checkpoints/dcgan/best_dcgan.pt",
    output_dir: str = "outputs/generated_samples",
    device: str = None,
    save_individuals: bool = True,
    grid_cols: int = 10,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Fallback to latest if best doesn't exist
    if not os.path.exists(ckpt_path):
        alt_path = ckpt_path.replace("best_", "latest_")
        if os.path.exists(alt_path):
            print(f"[Warning] '{ckpt_path}' not found, falling back to '{alt_path}'")
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading weights from: {ckpt_path}")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: GANConfig = checkpoint.get("config", default_gan_config)

    # Initialize generator with checkpoint configuration
    generator = Generator(
        z_dim=cfg.z_dim,
        channels=cfg.channels,
        ngf=getattr(cfg, "ngf", 64),
    ).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    # Load discriminator to evaluate realness score D(G(z))
    discriminator = None
    if "discriminator_state_dict" in checkpoint:
        try:
            discriminator = Discriminator(
                channels=cfg.channels,
                ndf=getattr(cfg, "ndf", 64),
                use_spectral_norm=getattr(cfg, "spectral_norm", True),
            ).to(device)
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            discriminator.eval()
        except Exception as e:
            print(f"[Warning] Could not load discriminator: {e}")
            discriminator = None

    os.makedirs(output_dir, exist_ok=True)
    print(f"Synthesizing {n} samples from latent prior N(0, I) (z_dim={cfg.z_dim})...")

    with torch.no_grad():
        # Latent sampling
        z = torch.randn(n, cfg.z_dim, device=device)
        fake_images = generator(z)
        fake_disp = denormalize(fake_images)

        # Discriminator scoring
        scores = None
        if discriminator is not None:
            raw_scores = discriminator(fake_images).squeeze()
            scores = raw_scores.cpu()
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            print(f"Discriminator average confidence D(G(z)): {scores.mean().item():.4f}")
            print(f"Score range: min={scores.min().item():.4f}, max={scores.max().item():.4f}")

        # Save all individual images
        if save_individuals:
            individual_dir = os.path.join(output_dir, "individuals")
            os.makedirs(individual_dir, exist_ok=True)
            for i in range(n):
                score_suffix = f"_score_{scores[i].item():.4f}" if scores is not None else ""
                file_name = f"sample_{i+1:03d}{score_suffix}.png"
                save_image(fake_disp[i], os.path.join(individual_dir, file_name))
            print(f"Saved {n} individual images to: {individual_dir}/")

        # Save full composite grid
        cols = min(grid_cols, n)
        grid = make_grid(fake_disp, nrow=cols, padding=2, normalize=False)
        grid_path = os.path.join(output_dir, f"grid_{n}_samples.png")
        save_image(grid, grid_path)
        print(f"Saved full {n}-image composite grid to: {grid_path}")

        # Extract and save Top-K realest images
        if scores is not None and top_k > 0:
            k = min(top_k, n)
            top_values, top_indices = torch.topk(scores, k=k, largest=True)

            top_dir = os.path.join(output_dir, f"top_{k}_realest")
            os.makedirs(top_dir, exist_ok=True)

            print(f"\nTop-{k} Realest Samples (ranked by D score):")
            print("-" * 55)
            for rank, (score_val, orig_idx) in enumerate(zip(top_values, top_indices), start=1):
                idx = orig_idx.item()
                val = score_val.item()
                top_img_path = os.path.join(top_dir, f"rank_{rank:02d}_sample_{idx+1:03d}_score_{val:.4f}.png")
                save_image(fake_disp[idx], top_img_path)
                print(f"  Rank #{rank:2d}: Sample {idx+1:03d} -> D Score = {val:.4f} (Saved: {top_img_path})")

            # Save Top-K grid
            top_images = fake_disp[top_indices]
            top_grid = make_grid(top_images, nrow=min(k, 6), padding=2, normalize=False)
            top_grid_path = os.path.join(output_dir, f"top_{k}_realest_grid.png")
            save_image(top_grid, top_grid_path)
            print("-" * 55)
            print(f"Saved Top-{k} composite grid to: {top_grid_path}")

    return grid_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate N samples from GAN and extract top realest images")
    parser.add_argument("--n", type=int, default=50, help="Number of images to generate (default: 50)")
    parser.add_argument("--top-k", type=int, default=6, help="Number of top realest images to select (default: 6)")
    parser.add_argument("--ckpt", type=str, default="checkpoints/dcgan/best_dcgan.pt", help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/generated_samples", help="Destination folder")
    parser.add_argument("--grid-cols", type=int, default=10, help="Columns in summary grid (default: 10)")
    parser.add_argument("--no-individuals", action="store_true", help="Do not save individual files, only the grid")

    args = parser.parse_args()
    generate_samples(
        n=args.n,
        top_k=args.top_k,
        ckpt_path=args.ckpt,
        output_dir=args.output_dir,
        save_individuals=not args.no_individuals,
        grid_cols=args.grid_cols,
    )
