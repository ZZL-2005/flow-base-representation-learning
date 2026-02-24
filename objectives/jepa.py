import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_crop_with_padding(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    b, c, h, w = x.shape
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    max_offset = 2 * pad
    top = torch.randint(0, max_offset + 1, (b,), device=x.device)
    left = torch.randint(0, max_offset + 1, (b,), device=x.device)
    out = torch.empty_like(x)
    for i in range(b):
        out[i] = x_pad[i, :, top[i] : top[i] + h, left[i] : left[i] + w]
    flip_mask = torch.rand(b, device=x.device) < 0.5
    out[flip_mask] = torch.flip(out[flip_mask], dims=[3])
    return out


class JEPAObjective(nn.Module):
    def __init__(self, backbone: nn.Module, momentum: float = 0.99):
        super().__init__()
        self.backbone = backbone
        self.target_backbone = copy.deepcopy(backbone)
        self.momentum = momentum
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        dim = backbone.embed_dim
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    @torch.no_grad()
    def update_ema(self) -> None:
        for p_online, p_target in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            p_target.data.mul_(self.momentum).add_(p_online.data * (1.0 - self.momentum))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_ctx = random_crop_with_padding(x)
        x_tgt = random_crop_with_padding(x)

        ctx_tokens, _ = self.backbone(x_ctx)
        ctx_repr = ctx_tokens.mean(dim=1)
        pred = self.predictor(ctx_repr)

        with torch.no_grad():
            tgt_tokens, _ = self.target_backbone(x_tgt)
            tgt_repr = tgt_tokens.mean(dim=1)

        loss = F.mse_loss(pred, tgt_repr)
        return {"loss": loss}
