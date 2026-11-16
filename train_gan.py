#!/usr/bin/env python3
"""
Main Training Script for Deep Convolutional GAN (DCGAN).
Trains a convolutional Generator and Discriminator on WikiArt images.
Configuration is loaded from gan_config.yaml (no argparse).
"""

import os
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gan_config import default_gan_config, GANConfig
from src.gan_model import Generator, Discriminator
from src.gan_dataset import get_gan_dataloaders


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Converts [-1, 1] normalized images to [0, 1] for visual display."""
    return torch.clamp(tensor * 0.5 + 0.5, 0.0, 1.0)


def train_one_epoch(
    G: nn.Module,
    D: nn.Module,
    train_loader: DataLoader,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    z_dim: int,
    label_smoothing: float = 0.0,
) -> Dict[str, float]:
    G.train()
    D.train()

    total_d_loss = 0.0
    total_d_loss_real = 0.0
    total_d_loss_fake = 0.0
    total_g_loss = 0.0
    total_d_real_score = 0.0
    total_d_fake_score = 0.0
    num_batches = 0

    # Label smoothing values
    real_label_val = 1.0 - label_smoothing  # e.g. 0.9
    fake_label_val = label_smoothing         # e.g. 0.1

    pbar = tqdm(train_loader, desc="Training DCGAN", leave=False)
    for real_images, _ in pbar:
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        # ====================================================================
        # 1. Train Discriminator
        # ====================================================================

        # Real images
        real_preds = D(real_images)
        real_labels = torch.full_like(real_preds, real_label_val, device=device)
        d_loss_real = criterion(real_preds, real_labels)

        # Fake images
        noise = torch.randn(batch_size, z_dim, device=device)
        fake_images = G(noise)
        fake_preds = D(fake_images.detach())
        fake_labels = torch.full_like(fake_preds, fake_label_val, device=device)
        d_loss_fake = criterion(fake_preds, fake_labels)

        # Update D
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ====================================================================
        # 2. Train Generator — FRESH noise, not reused D-step fakes
        # ====================================================================

        noise_g = torch.randn(batch_size, z_dim, device=device)
        fake_for_g = G(noise_g)
        fake_preds_for_g = D(fake_for_g)
        # Generator wants D to output 1.0 (real) for its fakes
        g_loss = criterion(fake_preds_for_g, torch.ones_like(fake_preds_for_g))

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Metrics tracking
        total_d_loss += d_loss.item()
        total_d_loss_real += d_loss_real.item()
        total_d_loss_fake += d_loss_fake.item()
        total_g_loss += g_loss.item()
        total_d_real_score += real_preds.mean().item()
        total_d_fake_score += fake_preds_for_g.mean().item()
        num_batches += 1

        pbar.set_postfix({
            "d_loss": f"{d_loss.item():.4f}",
            "g_loss": f"{g_loss.item():.4f}",
            "D(x)": f"{real_preds.mean().item():.2f}",
            "D(G(z))": f"{fake_preds_for_g.mean().item():.2f}",
        })

    n = max(num_batches, 1)
    return {
        "d_loss": total_d_loss / n,
        "d_loss_real": total_d_loss_real / n,
        "d_loss_fake": total_d_loss_fake / n,
        "g_loss": total_g_loss / n,
        "d_real_score": total_d_real_score / n,
        "d_fake_score": total_d_fake_score / n,
    }


def train_gan(cfg: GANConfig = default_gan_config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # DataLoader
    print("[Data Pipeline] Preparing dataset and DataLoaders...")
    train_loader = get_gan_dataloaders(
        dataset_name=cfg.dataset_name,
        data_dir=cfg.data_dir,
        kaggle_slug=cfg.kaggle_dataset_slug,
        img_size=cfg.img_size,
        channels=cfg.channels,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        download=True,
        max_gb=cfg.max_dataset_size_gb,
        max_images_per_artform=cfg.max_images_per_artform,
    )
    print(f"Dataset loaded with {len(train_loader.dataset)} images across {len(train_loader)} batches.")

    # Models — DCGAN architecture
    generator = Generator(
        z_dim=cfg.z_dim,
        channels=cfg.channels,
        ngf=cfg.ngf,
    ).to(device)
    discriminator = Discriminator(
        channels=cfg.channels,
        ndf=cfg.ndf,
        use_spectral_norm=cfg.spectral_norm,
    ).to(device)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator params:     {g_params:,}")
    print(f"Discriminator params: {d_params:,}")

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers — AdamW
    g_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    d_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    # LR Scheduler
    g_scheduler = None
    d_scheduler = None
    if cfg.scheduler == "cosine":
        g_scheduler = CosineAnnealingLR(g_optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
        d_scheduler = CosineAnnealingLR(d_optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
    elif cfg.scheduler == "step":
        g_scheduler = StepLR(g_optimizer, step_size=50, gamma=0.5)
        d_scheduler = StepLR(d_optimizer, step_size=50, gamma=0.5)

    # TensorBoard
    writer = SummaryWriter(log_dir=cfg.log_dir)
    print(f"TensorBoard logging to: {cfg.log_dir}")

    # Fixed noise for consistent visual monitoring
    fixed_noise = torch.randn(cfg.fixed_noise_samples, cfg.z_dim, device=device)

    print("\nStarting DCGAN Training:")
    print(f"  Architecture  : DCGAN (Conv)")
    print(f"  Dataset       : {cfg.dataset_name} ({cfg.kaggle_dataset_slug})")
    print(f"  Epochs        : {cfg.num_epochs}")
    print(f"  Batch size    : {cfg.batch_size}")
    print(f"  Image size    : {cfg.channels}×{cfg.img_size}×{cfg.img_size}")
    print(f"  Latent z_dim  : {cfg.z_dim}")
    print(f"  G features    : ngf={cfg.ngf}")
    print(f"  D features    : ndf={cfg.ndf}")
    print(f"  Optimizer     : AdamW (lr={cfg.lr}, betas=({cfg.beta1}, {cfg.beta2}), wd={cfg.weight_decay})")
    print(f"  Scheduler     : {cfg.scheduler}")
    print(f"  Label smooth  : {cfg.label_smoothing}")
    print(f"  Spectral norm : {cfg.spectral_norm}")
    print(f"  Loss          : BCELoss")
    print("-" * 70)

    best_g_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        metrics = train_one_epoch(
            G=generator,
            D=discriminator,
            train_loader=train_loader,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            criterion=criterion,
            device=device,
            z_dim=cfg.z_dim,
            label_smoothing=cfg.label_smoothing,
        )

        # Step LR schedulers
        current_lr = g_optimizer.param_groups[0]["lr"]
        if g_scheduler is not None:
            g_scheduler.step()
            d_scheduler.step()

        # TensorBoard Scalars
        writer.add_scalar("Loss/Discriminator_Total", metrics["d_loss"], epoch)
        writer.add_scalar("Loss/Discriminator_Real", metrics["d_loss_real"], epoch)
        writer.add_scalar("Loss/Discriminator_Fake", metrics["d_loss_fake"], epoch)
        writer.add_scalar("Loss/Generator", metrics["g_loss"], epoch)
        writer.add_scalar("Scores/D_Real_Accuracy", metrics["d_real_score"], epoch)
        writer.add_scalar("Scores/D_Fake_Accuracy", metrics["d_fake_score"], epoch)
        writer.add_scalar("LR/learning_rate", current_lr, epoch)

        # Visuals: Generate samples using fixed noise
        if epoch % cfg.sample_interval_epochs == 0 or epoch == cfg.num_epochs:
            generator.eval()
            with torch.no_grad():
                fake_samples = generator(fixed_noise)
                fake_disp = denormalize(fake_samples.cpu())
                grid = make_grid(fake_disp, nrow=4, normalize=False)
                writer.add_image("Visuals/Generated_Samples", grid, epoch)

                os.makedirs("outputs", exist_ok=True)
                sample_file = os.path.join("outputs", f"dcgan_samples_epoch_{epoch:03d}.png")
                save_image(grid, sample_file)

        # Checkpoints
        latest_path = os.path.join(cfg.checkpoint_dir, "latest_dcgan.pt")
        torch.save({
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg,
        }, latest_path)

        # Save best generator checkpoint
        if metrics["g_loss"] < best_g_loss:
            best_g_loss = metrics["g_loss"]
            best_path = os.path.join(cfg.checkpoint_dir, "best_dcgan.pt")
            torch.save({
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "metrics": metrics,
                "config": cfg,
            }, best_path)

        print(
            f"Epoch [{epoch:03d}/{cfg.num_epochs:03d}] "
            f"D Loss: {metrics['d_loss']:.4f} (R: {metrics['d_loss_real']:.4f}, F: {metrics['d_loss_fake']:.4f}) | "
            f"G Loss: {metrics['g_loss']:.4f} | "
            f"D(x): {metrics['d_real_score']:.3f} | "
            f"D(G(z)): {metrics['d_fake_score']:.3f} | "
            f"LR: {current_lr:.6f}"
        )

    writer.close()
    print("-" * 70)
    print(f"DCGAN Training completed! Best checkpoint: {os.path.join(cfg.checkpoint_dir, 'best_dcgan.pt')}")


if __name__ == "__main__":
    train_gan(default_gan_config)
