from typing import Optional

import torch
import torch.nn as nn

from models.vit_backbone import TransformerBlock


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = x.shape
    assert h % patch_size == 0 and w % patch_size == 0
    gh = h // patch_size
    gw = w // patch_size
    x = x.reshape(b, c, gh, patch_size, gw, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, patch_size * patch_size * c)
    return x


def unpatchify(patches: torch.Tensor, patch_size: int, img_size: int, channels: int = 3) -> torch.Tensor:
    b, n, d = patches.shape
    gh = gw = img_size // patch_size
    assert n == gh * gw and d == patch_size * patch_size * channels
    x = patches.reshape(b, gh, gw, patch_size, patch_size, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, channels, img_size, img_size)
    return x


class LightweightDecoder(nn.Module):
    def __init__(
        self,
        num_patches: int,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        patch_dim: int = 48,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=0.0,
                    drop_path=0.0,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_dim)

    def forward(self, x_tokens: torch.Tensor, token_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x_tokens
        if token_indices is not None:
            x = torch.gather(x, dim=1, index=token_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x)
