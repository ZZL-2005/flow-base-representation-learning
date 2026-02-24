from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import patchify
from models.time_embedding import SinusoidalTimeEmbedding


class FlowMatchingObjective(nn.Module):
    def __init__(self, backbone: nn.Module, patch_size: int = 4):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        patch_dim = 3 * patch_size * patch_size
        self.time_embed = SinusoidalTimeEmbedding(embed_dim=backbone.embed_dim)
        self.head = nn.Linear(backbone.embed_dim, patch_dim)

    def forward(self, x1: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(b, device=x1.device)
        xt = (1.0 - t).view(b, 1, 1, 1) * x0 + t.view(b, 1, 1, 1) * x1
        u = x1 - x0

        tokens = self.backbone.embed_patches(xt)
        temb = self.time_embed(t)
        hidden = self.backbone.forward_tokens(tokens, time_emb=temb)[-1]
        v_pred = self.head(hidden)

        target = patchify(u, self.patch_size)
        loss = F.mse_loss(v_pred, target)
        return {"loss": loss}
