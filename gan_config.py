import os
from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class GANConfig:
    # --- Dataset settings ---
    dataset_name: str = "wikiart"
    kaggle_dataset_slug: str = "steubk/wikiart"
    data_dir: str = "data/wikiart"
    img_size: int = 64
    channels: int = 3
    num_workers: int = 4
    max_dataset_size_gb: float = 5.0
    max_images_per_artform: int = 1000

    # --- Model Architecture (DCGAN) ---
    z_dim: int = 128
    ngf: int = 64          # Generator base feature map count
    ndf: int = 64          # Discriminator base feature map count
    spectral_norm: bool = True

    # --- Training Hyperparameters ---
    batch_size: int = 64
    num_epochs: int = 200
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    scheduler: str = "cosine"    # "cosine", "step", or "none"

    # --- Logging & Checkpoints ---
    log_dir: str = "runs/wikiart_dcgan"
    checkpoint_dir: str = "checkpoints/dcgan"
    sample_interval_epochs: int = 5
    fixed_noise_samples: int = 16

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


def load_gan_config(yaml_path: str = "gan_config.yaml") -> GANConfig:
    """
    Loads GAN configuration from a YAML file.
    Falls back to default parameters if the YAML file does not exist.
    """
    if not os.path.exists(yaml_path):
        return GANConfig()

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    dataset = data.get("dataset", {})
    model = data.get("model", {})
    training = data.get("training", {})
    logging_cfg = data.get("logging", {})

    return GANConfig(
        dataset_name=dataset.get("name", "wikiart"),
        kaggle_dataset_slug=dataset.get("kaggle_slug", "steubk/wikiart"),
        data_dir=dataset.get("data_dir", "data/wikiart"),
        img_size=dataset.get("img_size", 64),
        channels=dataset.get("channels", 3),
        num_workers=dataset.get("num_workers", 4),
        max_dataset_size_gb=float(dataset.get("max_dataset_size_gb", 5.0)),
        max_images_per_artform=int(dataset.get("max_images_per_artform", 1000)),
        z_dim=model.get("z_dim", 128),
        ngf=model.get("ngf", 64),
        ndf=model.get("ndf", 64),
        spectral_norm=model.get("spectral_norm", True),
        batch_size=training.get("batch_size", 64),
        num_epochs=training.get("num_epochs", 200),
        lr=float(training.get("lr", 2e-4)),
        beta1=float(training.get("beta1", 0.5)),
        beta2=float(training.get("beta2", 0.999)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        label_smoothing=float(training.get("label_smoothing", 0.1)),
        scheduler=training.get("scheduler", "cosine"),
        log_dir=logging_cfg.get("log_dir", "runs/wikiart_dcgan"),
        checkpoint_dir=logging_cfg.get("checkpoint_dir", "checkpoints/dcgan"),
        sample_interval_epochs=logging_cfg.get("sample_interval_epochs", 5),
        fixed_noise_samples=logging_cfg.get("fixed_noise_samples", 16),
    )


# Default configuration instance for GAN loaded from YAML
default_gan_config = load_gan_config("gan_config.yaml")
