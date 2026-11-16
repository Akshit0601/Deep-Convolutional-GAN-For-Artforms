# Deep Convolutional GAN (DCGAN) for WikiArt Image Generation

A PyTorch implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) trained on the Kaggle [WikiArt](https://www.kaggle.com/datasets/steubk/wikiart/data) dataset. Features convolutional generator and discriminator networks with spectral normalization, batch normalization, label smoothing, AdamW optimizer with Cosine Annealing learning rate scheduling, automated on-the-fly streaming dataset extraction, TensorBoard tracking, and sample synthesis utilities.
<p align="center">
<img width="662" height="332" alt="grid_50_samples" src="https://github.com/user-attachments/assets/d214a6d0-3f9c-47fd-8ba7-8b358f3fd0ac" />
   </p>

---

## 1. Environment Setup

### Prerequisites
* Python 3.12+
* NVIDIA GPU with CUDA drivers (e.g., RTX 4050 Laptop GPU)
* Kaggle account with API credentials

### Step-by-Step Setup

1. **Activate Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure Kaggle Credentials** *(Required for automatic WikiArt dataset streaming)*:
   Ensure `~/.kaggle/credentials.json` or `~/.kaggle/kaggle.json` exists, or set environment variables:
   ```bash
   export KAGGLE_USERNAME="your-username"
   export KAGGLE_KEY="your-api-key"
   ```

---

## 2. Workspace Structure

```
vae_setup/ (branch: gan)
├── .gitignore            # Git ignore rules
├── requirements.txt      # Project dependencies
├── gan_config.yaml       # GAN Hyperparameter & dataset configuration (YAML)
├── gan_config.py         # GAN Configuration dataclass & YAML loader
├── train_gan.py          # GAN Training pipeline, checkpointing & TensorBoard logging
├── sample_gan.py         # Sample synthesis & Top-K realest images extraction script
├── evaluate_gan.py       # GAN Generation & evaluation script
├── README.md             # Project documentation
│
├── src/
│   ├── __init__.py       # Package exports
│   ├── gan_model.py      # DCGAN architecture (Generator, Discriminator)
│   └── gan_dataset.py    # WikiArt dataset streaming loader & transforms
│
├── data/
│   └── wikiart/          # Extracted WikiArt images categorized by artform
├── checkpoints/
│   └── dcgan/            # Trained DCGAN checkpoints (best_dcgan.pt, latest_dcgan.pt)
├── outputs/              # Generated visual sample grids and individual images
│   └── generated_samples/# N-samples output and top realest extractions
└── runs/
    └── wikiart_dcgan/    # TensorBoard logs (Losses, D(x), D(G(z)), LR, visuals)
```

---

## 3. Configuration (`gan_config.yaml`)

Configuration is managed entirely through YAML (no `argparse` required for training):

```yaml
dataset:
  name: "wikiart"
  kaggle_slug: "steubk/wikiart"
  data_dir: "data/wikiart"
  img_size: 64
  channels: 3
  num_workers: 4
  max_dataset_size_gb: 5.0
  max_images_per_artform: 1000

model:
  z_dim: 128
  ngf: 64
  ndf: 64
  spectral_norm: true

training:
  batch_size: 64
  num_epochs: 200
  lr: 0.0002
  beta1: 0.5
  beta2: 0.999
  weight_decay: 0.0001
  label_smoothing: 0.1
  scheduler: "cosine"

logging:
  log_dir: "runs/wikiart_dcgan"
  checkpoint_dir: "checkpoints/dcgan"
  sample_interval_epochs: 5
  fixed_noise_samples: 16
```

---

## 4. Training the GAN

Launch training:
```bash
python train_gan.py
```

### TensorBoard Monitoring
Monitor training progress, discriminator/generator losses, real/fake scores, and visual samples in real time:
```bash
tensorboard --logdir runs/wikiart_dcgan
```

---

## 5. Generating Samples

Synthesize new artwork using the best saved checkpoint:

```bash
# Generate 50 samples and extract top 6 most realistic images
python sample_gan.py --n 50 --top-k 6

# Generate 100 samples with custom grid settings
python sample_gan.py --n 100 --top-k 10 --output-dir outputs/my_art
```
