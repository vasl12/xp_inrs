"""Minimal image-cohort dataset utilities for the Strainer SAE notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from skimage import io as skimage_io
from torch.utils.data import Dataset


def _resolve_relative_to_repo(path: str | Path) -> Path:
    """Resolve a path relative to this repo, with a fallback to the sibling data repo."""
    p = Path(path)
    if p.is_absolute():
        return p

    repo_root = Path(__file__).resolve().parent
    local = repo_root / p
    if local.exists():
        return local

    # Existing notebook outputs show the data living in ../sparse_inr/data.
    sibling_sparse_inr = repo_root.parent / "sparse_inr" / p
    if sibling_sparse_inr.exists():
        return sibling_sparse_inr

    return local


def resolve_cohort_image_paths(data_cfg) -> list[Path]:
    """Resolve numbered CelebA images for the notebook cohort experiment."""
    celeba_dir = _resolve_relative_to_repo(
        data_cfg.get("celeba_dir", "data/celeba/img_align_celeba")
    )
    if not celeba_dir.is_dir():
        raise FileNotFoundError(f"CelebA directory not found: {celeba_dir}")

    start = int(data_cfg.get("celeba_start", 1))
    max_images = int(data_cfg.get("max_images", 10))

    paths: list[Path] = []
    idx = start
    while len(paths) < max_images:
        fp = celeba_dir / f"{idx:06d}.jpg"
        if fp.is_file():
            paths.append(fp)
        idx += 1
        if idx - start > 500_000:
            raise FileNotFoundError(
                f"Could not collect {max_images} images under {celeba_dir} starting at {start}"
            )

    return paths


def _load_image_file(path: str | Path) -> np.ndarray:
    img = skimage_io.imread(path).astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img


class CohortDataset(Dataset):
    """A fixed set of same-shaped images represented as coordinate/pixel pairs."""

    def __init__(self, image_paths: list[str | Path]):
        self.paths = [Path(p) for p in image_paths]
        self.images = [
            torch.from_numpy(np.ascontiguousarray(_load_image_file(p))) for p in self.paths
        ]

        shapes = {tuple(im.shape) for im in self.images}
        if len(shapes) != 1:
            raise ValueError(f"CohortDataset requires identical image shapes, got {sorted(shapes)}")

        ref = self.images[0]
        h, w = ref.shape[:2]
        self.img_shape = (h, w)
        self.channels = 1 if ref.ndim == 2 else int(ref.shape[2])
        self.n_images = len(self.images)
        self.pixels_per_image = h * w

        xs = torch.linspace(0, 1, w)
        ys = torch.linspace(0, 1, h)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.coord_grid = torch.stack([xx, yy], dim=-1)
        self.coords = self.coord_grid.reshape(-1, 2)

        flats = []
        for image in self.images:
            if image.ndim == 2:
                flats.append(image.reshape(-1, 1))
            else:
                flats.append(image.reshape(-1, self.channels))
        self._flat_values = torch.stack(flats, dim=0)

    def __len__(self) -> int:
        return self.n_images * self.pixels_per_image

    def __getitem__(self, idx: int):
        image_idx = idx // self.pixels_per_image
        pixel_idx = idx % self.pixels_per_image
        return self.coords[pixel_idx], self._flat_values[image_idx, pixel_idx], int(image_idx)

    def get_image_tensor(self, image_index: int) -> torch.Tensor:
        return self.images[image_index]

    def flat_targets(self, image_index: int) -> torch.Tensor:
        return self._flat_values[image_index]
