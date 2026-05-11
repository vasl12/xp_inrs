"""TopK Sparse Autoencoder for INR feature analysis."""

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder.

    Learns an overcomplete dictionary of features from INR activations.
    Sparsity is enforced by keeping only the top-k activations after ReLU,
    rather than an L1 penalty.

    Parameters
    ----------
    d_model : int
        Dimension of the input activations (INR hidden dim).
    d_dict : int
        Dictionary size (number of SAE features, typically >> d_model).
    k : int
        Number of active features per input.
    """

    def __init__(self, d_model: int, d_dict: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_dict = d_dict
        self.k = k

        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.encoder = nn.Linear(d_model, d_dict, bias=True)
        self.decoder = nn.Linear(d_dict, d_model, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        """Project decoder weight columns to unit norm."""
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse codes.

        Returns
        -------
        z : (batch, d_dict) sparse tensor with exactly k nonzeros per row.
        """
        x_centered = x - self.pre_bias
        z_pre = self.encoder(x_centered)
        z_relu = torch.relu(z_pre)

        topk_vals, topk_idx = torch.topk(z_relu, self.k, dim=-1)
        z = torch.zeros_like(z_relu)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse codes back to activation space."""
        return self.decoder(z) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns (x_hat, z) where x_hat is the reconstruction
        and z is the sparse code.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(
        self,
        x: torch.Tensor,
        ortho_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Compute reconstruction loss and diagnostics.

        Parameters
        ----------
        x : torch.Tensor
            Input activations.
        ortho_weight : float
            Weight for orthogonal decoder penalty (0 = disabled).
            Penalizes off-diagonal entries of D^T D to encourage distinct features.

        Returns a dict with:
        - mse: mean squared reconstruction error
        - r2: coefficient of determination (1 = perfect reconstruction)
        - l0: average number of active features (should be ~k)
        - dead_frac: fraction of dictionary features that are never active
        - ortho_loss: decoder coherence penalty (if ortho_weight > 0)
        """
        x_hat, z = self.forward(x)
        ss_res = (x - x_hat).pow(2).sum()
        ss_tot = (x - x.mean(dim=0)).pow(2).sum()
        mse = (x - x_hat).pow(2).mean()
        r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-8)
        l0 = (z > 0).float().sum(dim=-1).mean()
        dead_frac = (z.sum(dim=0) == 0).float().mean()

        out = {"mse": mse, "r2": r2, "l0": l0, "dead_frac": dead_frac}

        if ortho_weight > 0:
            # D is (d_model, d_dict), columns are decoder vectors (unit norm).
            # D^T D has 1s on diagonal. Penalize off-diagonal = encourage orthogonality.
            D = self.decoder.weight  # (d_model, d_dict)
            gram = D.t() @ D  # (d_dict, d_dict)
            # Zero out diagonal, penalize squared off-diagonal
            mask = ~torch.eye(self.d_dict, device=gram.device, dtype=torch.bool)
            ortho_loss = (gram[mask] ** 2).mean()
            out["ortho_loss"] = ortho_loss
            out["loss"] = mse + ortho_weight * ortho_loss
        else:
            out["loss"] = mse

        return out
