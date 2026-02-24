from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import LightweightDecoder, unpatchify


class VAEObjective(nn.Module):
    def __init__(self, backbone: nn.Module, latent_dim: int = 256, beta: float = 1.0, img_size: int = 32, patch_size: int = 4):
        super().__init__()
        self.backbone = backbone
        self.beta = beta
        self.img_size = img_size
        self.patch_size = patch_size
        patch_dim = 3 * patch_size * patch_size
        self.mu_head = nn.Linear(backbone.embed_dim, latent_dim)
        self.logvar_head = nn.Linear(backbone.embed_dim, latent_dim)
        self.latent_to_tokens = nn.Linear(latent_dim, backbone.num_patches * backbone.embed_dim)
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
        pooled = tokens.mean(dim=1)
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        dec_tokens = self.latent_to_tokens(z).reshape(z.shape[0], self.backbone.num_patches, self.backbone.embed_dim)
        pred_patches = self.decoder(dec_tokens)
        recon = unpatchify(pred_patches, self.patch_size, self.img_size, channels=3)
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl
        return {"loss": loss, "recon_loss": recon_loss.detach(), "kl": kl.detach()}
