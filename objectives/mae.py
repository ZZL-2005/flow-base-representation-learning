from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import LightweightDecoder, patchify


def random_masking(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, n, d = x.shape
    len_keep = int(n * (1 - mask_ratio))
    noise = torch.rand(b, n, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d))
    mask = torch.ones(b, n, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


class MAEObjective(nn.Module):
    def __init__(self, backbone: nn.Module, mask_ratio: float = 0.75, img_size: int = 32, patch_size: int = 4):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        patch_dim = 3 * patch_size * patch_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, backbone.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder = LightweightDecoder(
            num_patches=backbone.num_patches,
            embed_dim=backbone.embed_dim,
            depth=4,
            num_heads=6,
            mlp_ratio=4.0,
            patch_dim=patch_dim,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_embed = self.backbone.embed_patches(x)
        x_vis, mask, ids_restore = random_masking(x_embed, self.mask_ratio)
        hidden_states = self.backbone.forward_tokens(x_vis)
        latent = hidden_states[-1]

        b = x.shape[0]
        n = self.backbone.num_patches
        len_keep = latent.shape[1]
        mask_tokens = self.mask_token.repeat(b, n - len_keep, 1)
        x_full = torch.cat([latent, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]))

        pred = self.decoder(x_full)
        target = patchify(x, self.patch_size)
        mse_per_patch = ((pred - target) ** 2).mean(dim=-1)
        loss = (mse_per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        return {"loss": loss}
