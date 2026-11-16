import os
from typing import Optional, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class WikiArtDataset(Dataset):
    """
    Dataset loader for WikiArt images.
    Recursively scans the directory for all valid image files.
    """
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths: List[str] = []

        if os.path.exists(root_dir):
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith(VALID_EXTENSIONS):
                        self.image_paths.append(os.path.join(root, f))

        self.image_paths.sort()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def get_gan_transforms(img_size: int = 64, channels: int = 3) -> transforms.Compose:
    """
    Returns image transformation pipeline for DCGAN:
    - Converts to Grayscale (1 channel) or RGB (3 channels)
    - Resizes to target resolution (64x64 for DCGAN)
    - Random horizontal flip for data augmentation
    - Converts PIL to FloatTensor [0, 1]
    - Normalizes to [-1, 1] to match Generator's Tanh() output
    """
    transform_list = []
    if channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    elif channels == 3:
        transform_list.append(transforms.Lambda(lambda img: img.convert("RGB")))

    transform_list.extend([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5] * channels,
            std=[0.5] * channels,
        ),
    ])
    return transforms.Compose(transform_list)


def _get_kaggle_auth_headers_and_auth() -> Tuple[dict, Optional[Tuple[str, str]]]:
    """Retrieves Kaggle authentication headers or basic auth credentials."""
    import json
    home = os.path.expanduser("~")
    creds_path = os.path.join(home, ".kaggle", "credentials.json")
    kaggle_json_path = os.path.join(home, ".kaggle", "kaggle.json")

    if os.path.exists(creds_path):
        try:
            with open(creds_path, "r") as f:
                data = json.load(f)
            if "access_token" in data:
                return {"Authorization": f"Bearer {data['access_token']}"}, None
        except Exception:
            pass

    if os.path.exists(kaggle_json_path):
        try:
            with open(kaggle_json_path, "r") as f:
                data = json.load(f)
            if "username" in data and "key" in data:
                return {}, (data["username"], data["key"])
        except Exception:
            pass

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return {}, (username, key)

    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}, None

    return {}, None


def download_wikiart_subset(
    data_dir: str = "data/wikiart",
    kaggle_slug: str = "steubk/wikiart",
    max_gb: float = 5.0,
    max_images_per_artform: int = 1000,
) -> int:
    """
    Downloads images from WikiArt across multiple art movements using on-the-fly zip streaming.
    Directly extracts image files from the incoming stream without saving the full 33.6 GB archive,
    stopping when the extracted dataset reaches max_gb (e.g. 5.0 GB), with a per-artform cap
    to ensure wide artistic diversity across genres.
    """
    import struct
    import zlib
    import requests
    from tqdm import tqdm

    max_bytes = int(max_gb * 1024 * 1024 * 1024)

    # Calculate existing size and per-artform counts
    total_bytes = 0
    artform_counts = {}
    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(VALID_EXTENSIONS):
                    p = os.path.join(root, f)
                    total_bytes += os.path.getsize(p)
                    rel = os.path.relpath(p, data_dir)
                    artform = rel.split(os.sep)[0]
                    artform_counts[artform] = artform_counts.get(artform, 0) + 1

    if total_bytes >= max_bytes:
        print(f"[Data Pipeline] Local dataset already at or exceeding {max_gb:.1f} GB ({total_bytes / (1024**3):.2f} GB).")
        return sum(artform_counts.values())

    headers, auth = _get_kaggle_auth_headers_and_auth()
    url = f"https://www.kaggle.com/api/v1/datasets/download/{kaggle_slug}"

    print(f"\n[Data Pipeline] Streaming WikiArt images from Kaggle (capped at {max_gb:.1f} GB, max {max_images_per_artform} per artform)...")
    resp = requests.get(url, headers=headers, auth=auth, stream=True, allow_redirects=True)
    if resp.status_code != 200:
        raise RuntimeError(f"Kaggle download failed with status {resp.status_code}: {resp.text[:200]}")

    class StreamBuffer:
        def __init__(self, it):
            self.it = it
            self.buf = bytearray()

        def read(self, n):
            while len(self.buf) < n:
                try:
                    chunk = next(self.it)
                    self.buf.extend(chunk)
                except StopIteration:
                    break
            res = bytes(self.buf[:n])
            del self.buf[:n]
            return res

    sb = StreamBuffer(resp.iter_content(chunk_size=131072))
    os.makedirs(data_dir, exist_ok=True)
    new_extracted = 0

    pbar = tqdm(
        total=max_bytes,
        initial=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Streaming WikiArt (~5GB)",
        leave=True,
    )
    try:
        while total_bytes < max_bytes:
            sig = sb.read(4)
            if sig != b"PK\x03\x04":
                break

            header = sb.read(26)
            if len(header) < 26:
                break
            (ver, flags, method, mtime, mdate, crc32, comp_size, uncomp_size, fname_len, extra_len) = struct.unpack(
                "<HHHHHIIIHH", header
            )
            fname = sb.read(fname_len).decode("utf-8", errors="ignore")
            extra = sb.read(extra_len)

            # Support ZIP64 format for 64-bit sizes
            if comp_size == 0xFFFFFFFF or uncomp_size == 0xFFFFFFFF:
                off = 0
                while off + 4 <= len(extra):
                    tag, sz = struct.unpack("<HH", extra[off:off+4])
                    if tag == 0x0001:
                        uncomp_size, comp_size = struct.unpack("<QQ", extra[off+4:off+4+16])
                        break
                    off += 4 + sz

            comp_data = sb.read(comp_size)

            # Skip non-images
            if not fname.lower().endswith(VALID_EXTENSIONS):
                continue

            artform = fname.split("/")[0] if "/" in fname else "General"

            # Check per-artform cap to promote artistic diversity across styles
            if max_images_per_artform and artform_counts.get(artform, 0) >= max_images_per_artform:
                continue

            out_path = os.path.join(data_dir, fname)
            if os.path.exists(out_path):
                continue

            if method == 8:
                decomp = zlib.decompressobj(-15).decompress(comp_data)
            else:
                decomp = comp_data

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(decomp)

            file_size = len(decomp)
            total_bytes += file_size
            artform_counts[artform] = artform_counts.get(artform, 0) + 1
            new_extracted += 1
            pbar.update(file_size)
            pbar.set_postfix({"artforms": len(artform_counts), "images": sum(artform_counts.values())})
    finally:
        resp.close()
        pbar.close()

    total_images = sum(artform_counts.values())
    print(f"\n[Data Pipeline] Completed: {total_images} images across {len(artform_counts)} artforms ({total_bytes / (1024**3):.2f} GB) in '{data_dir}'.")
    return total_images


def ensure_wikiart_dataset(
    data_dir: str = "data/wikiart",
    kaggle_slug: str = "steubk/wikiart",
    download_subset_styles: Optional[List[str]] = None,
    max_gb: float = 5.0,
    max_images_per_artform: int = 1000,
) -> str:
    """
    Checks if WikiArt dataset exists locally in data_dir.
    If not, downloads images up to max_gb via the streaming API.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Check if images already exist
    existing_images = 0
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(VALID_EXTENSIONS):
                existing_images += 1
                if existing_images >= 50:
                    return data_dir

    print(f"\n[Data Pipeline] WikiArt images not found in '{data_dir}'. Starting download (capped at {max_gb:.1f} GB) from '{kaggle_slug}'...")

    try:
        download_wikiart_subset(data_dir=data_dir, kaggle_slug=kaggle_slug, max_gb=max_gb, max_images_per_artform=max_images_per_artform)
        print(f"[Data Pipeline] Successfully prepared WikiArt dataset in '{data_dir}'.")
    except Exception as e:
        print(f"[Data Pipeline Warning] Automatic Kaggle download encountered an issue: {e}")
        print("Falling back to local checking or MNIST if no local images are present.")

    return data_dir


def get_gan_dataloaders(
    dataset_name: str = "wikiart",
    data_dir: str = "data/wikiart",
    kaggle_slug: str = "steubk/wikiart",
    img_size: int = 28,
    channels: int = 1,
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
    max_gb: float = 5.0,
    max_images_per_artform: int = 1000,
) -> DataLoader:
    """
    Builds DataLoader for GAN training.
    Supports WikiArt with automatic download via Kaggle,
    with seamless fallback to MNIST if WikiArt has not yet been populated.
    """
    transform = get_gan_transforms(img_size=img_size, channels=channels)

    if dataset_name.lower() == "wikiart":
        if download:
            ensure_wikiart_dataset(
                data_dir=data_dir,
                kaggle_slug=kaggle_slug,
                max_gb=max_gb,
                max_images_per_artform=max_images_per_artform,
            )

        dataset = WikiArtDataset(root_dir=data_dir, transform=transform)

        if len(dataset) == 0:
            print(f"[Data Pipeline Warning] No WikiArt images found in '{data_dir}'. Using MNIST for demonstration/testing.")
            dataset = datasets.MNIST(
                root="data",
                train=True,
                download=True,
                transform=transform,
            )
    elif dataset_name.lower() == "mnist":
        dataset = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )
    else:
        # Generic ImageFolder or custom directory
        dataset = WikiArtDataset(root_dir=data_dir, transform=transform)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    return train_loader
