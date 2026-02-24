import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_backbone


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR10(root=data_cfg["root"], train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_cfg["root"], train=False, download=True, transform=test_tf)
    bs = data_cfg["batch_size"]
    nw = data_cfg.get("num_workers", 4)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_loaders(cfg["data"])

    backbone = build_backbone(cfg["backbone"]).to(device)
    state = torch.load(args.backbone, map_location=device)
    backbone.load_state_dict(state)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    depth = cfg["backbone"]["depth"]
    probes = nn.ModuleList([nn.Linear(cfg["backbone"]["embed_dim"], 10) for _ in range(depth)]).to(device)
    optimizer = SGD(probes.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    best_layer_acc = [0.0 for _ in range(depth)]
    epochs = cfg["eval"]["linear_epochs"]

    for epoch in range(epochs):
        probes.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, hs = backbone(x, return_hidden_states=True)
                feats = [h.mean(dim=1) for h in hs]
            losses = [criterion(probes[i](feats[i]), y) for i in range(depth)]
            loss = torch.stack(losses).sum()
            loss.backward()
            optimizer.step()

        probes.eval()
        correct = [0 for _ in range(depth)]
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                _, hs = backbone(x, return_hidden_states=True)
                feats = [h.mean(dim=1) for h in hs]
                for i in range(depth):
                    pred = probes[i](feats[i]).argmax(dim=1)
                    correct[i] += (pred == y).sum().item()
                total += y.size(0)
        accs = [100.0 * c / total for c in correct]
        best_layer_acc = [max(best_layer_acc[i], accs[i]) for i in range(depth)]
        print(f"epoch={epoch+1}/{epochs} layer_last_acc={accs[-1]:.2f}")

    best_layer = max(best_layer_acc)
    last_layer = best_layer_acc[-1]
    out = {
        "best_layer_accuracy": best_layer,
        "last_layer_accuracy": last_layer,
        "per_layer_best_accuracy": best_layer_acc,
    }
    out_dir = Path(cfg["eval"].get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_name = cfg.get("name", cfg["objective"]["name"])
    out_path = out_dir / f"{exp_name}_linear_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved_linear_eval={out_path}")


if __name__ == "__main__":
    main()
