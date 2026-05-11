"""Minimal Strainer network used by ``strainer_sae_decomposition.ipynb``."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _alpine_linear_layer(
    in_features: int,
    out_features: int,
    omega: float = 30.0,
    bias: bool = True,
    is_first: bool = False,
    is_last: bool = False,
) -> nn.Linear:
    """SIREN-style linear initialization matching the Alpine Strainer notebook."""
    layer = nn.Linear(in_features, out_features, bias=bias)
    with torch.no_grad():
        if is_first:
            layer.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
        else:
            bound = np.sqrt(6.0 / in_features) / omega
            layer.weight.uniform_(-bound, bound)

        if is_last:
            last_bound = np.sqrt(6.0 / in_features) / max(omega, 1e-12)
            layer.weight.uniform_(-last_bound, last_bound)
    return layer


class _AlpineSine(nn.Module):
    def __init__(self, omega: float = 30.0):
        super().__init__()
        self.omega = float(omega)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)


class AlpineStrainer(nn.Module):
    """Shared SIREN-like encoder with one decoder tail per image."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        num_shared_layers: int,
        num_decoders: int,
        omegas: list[float] | None = None,
        bias: bool = True,
    ):
        super().__init__()
        if hidden_layers <= num_shared_layers:
            raise ValueError("hidden_layers must be greater than num_shared_layers")

        if omegas is None:
            omegas = [30.0]
        self._omegas = (
            [float(x) for x in omegas]
            if len(omegas) == hidden_layers
            else [float(omegas[0])] * hidden_layers
        )

        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.hidden_layers = int(hidden_layers)
        self.out_features = int(out_features)
        self.num_shared_layers = int(num_shared_layers)
        self.num_decoders = int(num_decoders)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(self.num_shared_layers):
            self.encoder.append(
                _alpine_linear_layer(
                    self.in_features if i == 0 else self.hidden_features,
                    self.hidden_features,
                    omega=self._omegas[i],
                    bias=bias,
                    is_first=(i == 0),
                )
            )
            self.encoder.append(_AlpineSine(self._omegas[i]))

        for _ in range(self.num_decoders):
            layers: list[nn.Module] = []
            for j in range(self.num_shared_layers, self.hidden_layers):
                is_last = j == self.hidden_layers - 1
                layers.append(
                    _alpine_linear_layer(
                        self.hidden_features,
                        self.out_features if is_last else self.hidden_features,
                        omega=self._omegas[j],
                        bias=bias,
                        is_last=is_last,
                    )
                )
                if not is_last:
                    layers.append(_AlpineSine(self._omegas[j]))
            self.decoder.append(nn.ModuleList(layers))

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode_index(self, enc_output: torch.Tensor, decoder_idx: int) -> torch.Tensor:
        x = enc_output.clone()
        for layer in self.decoder[decoder_idx]:
            x = layer(x)
        return x

    def forward(self, coords: torch.Tensor, return_features: bool = False) -> dict[str, torch.Tensor]:
        if return_features:
            raise NotImplementedError("return_features is not implemented for AlpineStrainer")
        encoded = self.encode(coords)
        outputs = [self.decode_index(encoded, i) for i in range(self.num_decoders)]
        return {"output": torch.stack(outputs, dim=1)}

    def load_encoder_weights(self, weights) -> None:
        self.encoder.load_state_dict(weights)
