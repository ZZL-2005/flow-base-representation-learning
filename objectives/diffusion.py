import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import patchify
from models.time_embedding import SinusoidalTimeEmbedding


def cosine_alpha_sigma(t: torch.Tensor, s: float = 0.008) -> Tuple[torch.Tensor, torch.Tensor]:
    f = torch.cos(((t + s) / (1.0 + s)) * math.pi * 0.5) ** 2
    alpha = f.sqrt()
    sigma = (1.0 - f).clamp(min=1e-8).sqrt()
    return alpha, sigma


class DiffusionObjective(nn.Module):
    def __init__(self, backbone: nn.Module, patch_size: int = 4):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        patch_dim = 3 * patch_size * patch_size
        self.time_embed = SinusoidalTimeEmbedding(embed_dim=backbone.embed_dim)
        self.head = nn.Linear(backbone.embed_dim, patch_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = x.shape[0]
        t = torch.rand(b, device=x.device)
        eps = torch.randn_like(x)
        alpha, sigma = cosine_alpha_sigma(t)
        xt = alpha.view(b, 1, 1, 1) * x + sigma.view(b, 1, 1, 1) * eps

        tokens = self.backbone.embed_patches(xt)
        temb = self.time_embed(t)
        hidden = self.backbone.forward_tokens(tokens, time_emb=temb)[-1]
        eps_pred = self.head(hidden)

        target = patchify(eps, self.patch_size)
        loss = F.mse_loss(eps_pred, target)
        return {"loss": loss}
