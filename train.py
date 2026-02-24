import argparse
import json
import math
import random
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_backbone
from objectives import build_objective


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_loaders(data_cfg: dict):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = datasets.CIFAR10(root=data_cfg["root"], train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_cfg["root"], train=False, download=True, transform=test_tf)
    batch_size = data_cfg["batch_size"]
    workers = data_cfg.get("num_workers", 4)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader


def cosine_warmup_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = build_loaders(cfg["data"])

    backbone = build_backbone(cfg["backbone"]).to(device)
    model = build_objective(cfg["objective"]["name"], backbone, cfg["objective"]).to(device)

    optim_cfg = cfg["optim"]
    optimizer = AdamW(
        model.parameters(),
        lr=optim_cfg["lr"],
        weight_decay=optim_cfg["weight_decay"],
    )

    epochs = cfg["train"]["epochs"]
    total_steps = epochs * len(train_loader)
    warmup_steps = cfg["train"]["warmup_epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cosine_warmup_lambda(step, total_steps, warmup_steps)
    )

    log = {
        "objective": cfg["objective"]["name"],
        "seed": cfg.get("seed", 42),
        "backbone_params": count_parameters(backbone),
        "total_params": count_parameters(model),
        "train_loss": [],
    }

    model.train()
    start = time.time()
    global_step = 0
    best_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    best_backbone_state = None
    for epoch in range(epochs):
        running = 0.0
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["train"]["grad_clip"])
            optimizer.step()
            scheduler.step()
            if cfg["objective"]["name"].lower() == "jepa":
                model.update_ema()
            running += loss.item()
            global_step += 1

        avg = running / len(train_loader)
        log["train_loss"].append(avg)
        print(f"epoch={epoch+1}/{epochs} loss={avg:.6f} lr={scheduler.get_last_lr()[0]:.6e}")
        if avg < best_loss:
            best_loss = avg
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            best_backbone_state = copy.deepcopy(backbone.state_dict())

    elapsed = time.time() - start
    log["train_time_sec"] = elapsed

    out_dir = Path(cfg["train"].get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_name = cfg.get("name", cfg["objective"]["name"])

    best_full_path = out_dir / f"{exp_name}_best_full.pt"
    best_backbone_path = out_dir / f"{exp_name}_best_backbone.pt"
    last_full_path = out_dir / f"{exp_name}_last_full.pt"
    last_backbone_path = out_dir / f"{exp_name}_last_backbone.pt"

    if best_model_state is not None:
        torch.save(best_model_state, best_full_path)
    if best_backbone_state is not None:
        torch.save(best_backbone_state, best_backbone_path)
    torch.save(model.state_dict(), last_full_path)
    torch.save(backbone.state_dict(), last_backbone_path)

    log["best_epoch"] = best_epoch
    log["best_train_loss"] = best_loss
    log["best_full_ckpt"] = str(best_full_path)
    log["best_backbone_ckpt"] = str(best_backbone_path)
    log["last_full_ckpt"] = str(last_full_path)
    log["last_backbone_ckpt"] = str(last_backbone_path)

    log_path = out_dir / f"{exp_name}_train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"saved_best_full={best_full_path}")
    print(f"saved_best_backbone={best_backbone_path}")
    print(f"saved_last_full={last_full_path}")
    print(f"saved_last_backbone={last_backbone_path}")
    print(f"saved_log={log_path}")


if __name__ == "__main__":
    main()
