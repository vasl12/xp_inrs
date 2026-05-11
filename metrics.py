"""Small image metrics used by the Strainer SAE notebook."""

from __future__ import annotations

import torch


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB for images/tensors in [0, 1]."""
    mse = torch.mean((pred - target) ** 2)
    return -10.0 * torch.log10(mse.clamp(min=1e-12))


def _gaussian_kernel(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] @ g[None, :]
    return kernel[None, None]


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Structural similarity for grayscale ``(H, W)`` or channel-last ``(H, W, C)`` images."""
    if pred.dim() == 2:
        pred = pred[None, None]
        target = target[None, None]
    elif pred.dim() == 3:
        pred = pred.permute(2, 0, 1)[None]
        target = target.permute(2, 0, 1)[None]
    else:
        raise ValueError(f"Expected 2D or 3D image tensors, got {tuple(pred.shape)}")

    kernel = _gaussian_kernel(window_size, 1.5, pred.device, pred.dtype)
    kernel = kernel.expand(pred.shape[1], -1, -1, -1)
    padding = window_size // 2

    mu_pred = torch.nn.functional.conv2d(pred, kernel, padding=padding, groups=pred.shape[1])
    mu_target = torch.nn.functional.conv2d(target, kernel, padding=padding, groups=target.shape[1])

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = (
        torch.nn.functional.conv2d(pred ** 2, kernel, padding=padding, groups=pred.shape[1])
        - mu_pred_sq
    )
    sigma_target_sq = (
        torch.nn.functional.conv2d(target ** 2, kernel, padding=padding, groups=target.shape[1])
        - mu_target_sq
    )
    sigma_cross = (
        torch.nn.functional.conv2d(pred * target, kernel, padding=padding, groups=pred.shape[1])
        - mu_cross
    )

    numerator = (2 * mu_cross + C1) * (2 * sigma_cross + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    return (numerator / denominator).mean()
