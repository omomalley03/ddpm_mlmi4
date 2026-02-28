# DDPM — Denoising Diffusion Probabilistic Models

PyTorch reimplementation of [Ho et al. 2020](https://arxiv.org/abs/2006.11239), trained on CIFAR-10 (32×32).

## Files

| File | Description |
|---|---|
| `model.py` | U-Net noise prediction network (~35M params) |
| `diffusion.py` | Forward/reverse diffusion process, loss, sampling |
| `dataset.py` | CIFAR-10 data loading (auto-downloaded) |
| `train.py` | Training loop with Adam + EMA |
| `sample.py` | Generate images from a checkpoint |
| `run.py` | Entry point (argparse) |
| `slurm_ddpm.sh` | Full training job (Wilkes3 HPC) |
| `slurm_test.sh` | Validation jobs: overfit test + short run |

## Requirements

```bash
pip install -r requirements.txt
```

Requires PyTorch ≥ 2.0 and a CUDA GPU for full training. CIFAR-10 (~170MB) is downloaded automatically on first run.

## Training

**Full training run** (1.3M steps, ~36-48h on A100):
```bash
python run.py --mode train
```

**Resume from checkpoint:**
```bash
python run.py --mode train --resume checkpoints/ckpt_500000.pt
```

## Sampling

```bash
python run.py --mode sample --resume checkpoints/ckpt_1300000.pt --n_samples 64
```

Saves a grid image and individual samples to `samples/`.

## Validation (before full training)

**Overfit test** (~30 min) — model must learn to reproduce a small set of images:
```bash
python run.py --mode train --total_steps 50000 --batch_size 32 --subset_size 512 \
    --save_dir checkpoints_overfit --save_every 10000 --log_every 500
python run.py --mode sample --resume checkpoints_overfit/ckpt_50000.pt --n_samples 16
```

**Short run** (~6-8h) — checks training dynamics on full data:
```bash
python run.py --mode train --total_steps 200000 --save_dir checkpoints_short \
    --save_every 10000 --log_every 1000
```

### Expected loss values

| Step | Expected loss |
|---|---|
| ~100 | 0.9–1.1 |
| 50k | 0.1–0.3 |
| 200k | 0.05–0.1 |
| 1.3M | ~0.03–0.05 |

## HPC (Wilkes3)

**Validation test:**
```bash
sbatch slurm_test.sh
```

**Full training:**
```bash
sbatch slurm_ddpm.sh
```

Monitor logs: `tail -f logs/out.$JOBID`

## Architecture

- **U-Net** backbone from PixelCNN++ (Wide ResNet style)
- Pre-activation residual blocks: GroupNorm → SiLU → Conv
- Sinusoidal timestep embeddings, broadcast to every ResBlock
- Single-head self-attention at 16×16 resolution only
- 4 resolution levels (32→16→8→4), 2 ResBlocks per level, base channels 128
- Linear β schedule: β₁=1e-4 to β_T=0.02, T=1000 steps
- Fixed sampling variance σ²=β_t
- EMA decay 0.9999, Adam lr=2e-4, dropout 0.1

## Target results (CIFAR-10)

| Metric | Value |
|---|---|
| FID | ~3.17 |
| Inception Score | ~9.46 |
| Training steps | 1.3M |
