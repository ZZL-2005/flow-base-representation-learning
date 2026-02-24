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
python eval_linear.py --config configs/ae.yaml --backbone outputs/ae_backbone.pt
```

Evaluate full fine-tuning:

```bash
python eval_finetune.py --config configs/ae.yaml --backbone outputs/ae_backbone.pt
```

Repeat eval for each objective checkpoint (`vae_backbone.pt`, `mae_backbone.pt`, etc.).
