from typing import Dict

from objectives.ae import AEObjective
from objectives.diffusion import DiffusionObjective
from objectives.flow_ot import FlowMatchingObjective
from objectives.jepa import JEPAObjective
from objectives.mae import MAEObjective
from objectives.vae import VAEObjective


def build_objective(name: str, backbone, cfg: Dict):
    name = name.lower()
    if name == "ae":
        return AEObjective(backbone, img_size=cfg.get("img_size", 32), patch_size=cfg.get("patch_size", 4))
    if name == "vae":
        return VAEObjective(
            backbone,
            latent_dim=cfg.get("latent_dim", 256),
            beta=cfg.get("beta", 1.0),
            img_size=cfg.get("img_size", 32),
            patch_size=cfg.get("patch_size", 4),
        )
    if name == "mae":
        return MAEObjective(
            backbone,
            mask_ratio=cfg.get("mask_ratio", 0.75),
            img_size=cfg.get("img_size", 32),
            patch_size=cfg.get("patch_size", 4),
        )
    if name == "jepa":
        return JEPAObjective(backbone, momentum=cfg.get("momentum", 0.99))
    if name == "diffusion":
        return DiffusionObjective(backbone, patch_size=cfg.get("patch_size", 4))
    if name == "flow":
        return FlowMatchingObjective(backbone, patch_size=cfg.get("patch_size", 4))
    raise ValueError(f"Unsupported objective: {name}")
