# DDPM — Denoising Diffusion Probabilistic Models

Reimplementation of [Ho et al. 2020](https://arxiv.org/abs/2006.11239), trained on CIFAR-10 (32x32), OAM/Vortex laser beam dataset, and CelebA-HQ.

Branch ``main'' looks at CIFAR and laser dataset.

**Note:** Most of these experiments can be accessed through `run.py` and setting the `--mode` flag to the file name to easily replicate slurm scripts.

## Experiment 1: CIFAR Replication

Files: `train.py`, `sample.py`, `eval.py`

## Experiment 2: DDPM to generate 128x128 laser images

Files: `train_ddpm_oam.py`, `sample_oam.py`

## Experiment 3: DDPM to generate latent representation of laser images

**Part A:** Train VAE on laser images.  
Files: `train_vae_oam.py`, `train_ddpm_latent.py`

**Part B:** Train DDPM to generate VAE mean vectors.  
Files: `train_ddpm_latent.py`, `sample_ldm.py` (this generates samples from pixel DDPM *and* latent DDPM to compare visually)

## Experiment 4: Scoring physical plausibility of generated OAM beams

Since FID score is not relevant to OAM beams, we train a CNN classifier on real images and test on generated images. Must train DDPMs on images of one OAM mode and one turb level in order to know what the DDPM is trying to sample, since it is unconditional.

**Training and Evaluating CNN**  
Files: `cnn_turb_classifier.py`. To evaluate, set `eval_dir` to the generated samples.


