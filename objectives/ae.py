from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import LightweightDecoder, unpatchify


class AEObjective(nn.Module):
    def __init__(self, backbone: nn.Module, img_size: int = 32, patch_size: int = 4):
        super().__init__()
        self.backbone = backbone
        self.img_size = img_size
        self.patch_size = patch_size
        patch_dim = 3 * patch_size * patch_size
        self.decoder = LightweightDecoder(
            num_patches=backbone.num_patches,
            embed_dim=backbone.embed_dim,
            depth=4,
            num_heads=6,
            mlp_ratio=4.0,
            patch_dim=patch_dim,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens, _ = self.backbone(x)
        pred_patches = self.decoder(tokens)
        recon = unpatchify(pred_patches, self.patch_size, self.img_size, channels=3)
        loss = F.mse_loss(recon, x)
        return {"loss": loss}
