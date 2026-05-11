"""Minimal layer-transfer helper used by ``strainer_sae_decomposition.ipynb``."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from network import _alpine_linear_layer


def _reinit_alpine_encoder_block(strainer: nn.Module, block_idx: int) -> None:
    linear = strainer.encoder[2 * block_idx]
    if not isinstance(linear, nn.Linear):
        raise TypeError("expected Linear at encoder even index")

    new_linear = _alpine_linear_layer(
        int(strainer.in_features) if block_idx == 0 else int(strainer.hidden_features),
        int(strainer.hidden_features),
        omega=float(strainer._omegas[block_idx]),
        bias=linear.bias is not None,
        is_first=(block_idx == 0),
    )
    with torch.no_grad():
        linear.weight.copy_(new_linear.weight.to(linear.weight))
        if linear.bias is not None and new_linear.bias is not None:
            linear.bias.copy_(new_linear.bias.to(linear.bias))


def _reinit_alpine_decoder(strainer: nn.Module) -> None:
    global_layer_idx = int(strainer.num_shared_layers)
    for module in strainer.decoder[0]:
        if not isinstance(module, nn.Linear):
            continue
        is_last = global_layer_idx == int(strainer.hidden_layers) - 1
        new_linear = _alpine_linear_layer(
            int(strainer.hidden_features),
            int(strainer.out_features) if is_last else int(strainer.hidden_features),
            omega=float(strainer._omegas[global_layer_idx]),
            bias=module.bias is not None,
            is_last=is_last,
        )
        with torch.no_grad():
            module.weight.copy_(new_linear.weight.to(module.weight))
            if module.bias is not None and new_linear.bias is not None:
                module.bias.copy_(new_linear.bias.to(module.bias))
        global_layer_idx += 1


def apply_l_freeze(
    strainer: nn.Module,
    l_freeze: int,
    *,
    reinit_seed: int,
    is_ffmlp: bool,
) -> None:
    """Freeze encoder blocks through ``l_freeze`` and reinitialize the trainable tail.

    The notebook only uses the Alpine/SIREN Strainer path. ``l_freeze=-1`` is the
    scratch baseline: all encoder blocks and the decoder are reinitialized.
    """
    if is_ffmlp:
        raise NotImplementedError("This notebook cleanup only keeps AlpineStrainer transfer")

    n_blocks = len(strainer.encoder) // 2
    if l_freeze < -1 or l_freeze >= n_blocks:
        raise ValueError(f"l_freeze must be -1 or 0..{n_blocks - 1}, got {l_freeze}")

    torch.manual_seed(int(reinit_seed))
    np.random.seed(int(reinit_seed) % (2**32 - 1))

    if l_freeze == -1:
        for param in strainer.parameters():
            param.requires_grad_(True)
        for block_idx in range(n_blocks):
            _reinit_alpine_encoder_block(strainer, block_idx)
        _reinit_alpine_decoder(strainer)
        return

    cut = 2 * (int(l_freeze) + 1)
    for idx, module in enumerate(strainer.encoder):
        for param in module.parameters():
            param.requires_grad_(idx >= cut)

    for param in strainer.decoder[0].parameters():
        param.requires_grad_(True)

    for block_idx in range(int(l_freeze) + 1, n_blocks):
        _reinit_alpine_encoder_block(strainer, block_idx)
    _reinit_alpine_decoder(strainer)
