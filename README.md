# Generative Objectives as Representation Learning (CIFAR-10)

Unified CIFAR-10 benchmark comparing `AE`, `VAE`, `MAE`, `JEPA`, `Diffusion (DDPM eps-pred)`, and `Flow Matching (OT path)` with the same ViT backbone.

## Run

Train one objective:

```bash
python train.py --config configs/ae.yaml
python train.py --config configs/vae.yaml
python train.py --config configs/mae.yaml
python train.py --config configs/jepa.yaml
python train.py --config configs/diffusion.yaml
python train.py --config configs/flow.yaml
```

Evaluate linear probes (all transformer blocks):

```bash
python eval_linear.py --config configs/ae.yaml --backbone outputs/ae_best_backbone.pt
```

Evaluate full fine-tuning:

```bash
python eval_finetune.py --config configs/ae.yaml --backbone outputs/ae_best_backbone.pt
```

Repeat eval for each objective checkpoint (`vae_best_backbone.pt`, `mae_best_backbone.pt`, etc.).

## Checkpoints

Each run now saves four files in `outputs/`:

- `{name}_best_full.pt`: best full objective model (backbone + objective-specific modules)
- `{name}_best_backbone.pt`: best backbone-only checkpoint (for linear probe / finetune)
- `{name}_last_full.pt`: last-epoch full objective model
- `{name}_last_backbone.pt`: last-epoch backbone-only checkpoint
