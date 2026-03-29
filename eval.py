"""
Quantitative evaluation for DDPM: FID and Inception Score.

Compares generated samples against real CIFAR-10 training images using
InceptionV3 features, matching the evaluation protocol from Ho et al. 2020.

Paper targets (CIFAR-10):
    FID: 3.17  (lower is better)
    IS:  9.46 ± 0.11  (higher is better)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from diffusion import GaussianDiffusion
from model import UNet


# ---------------------------------------------------------------------------
# Inception helpers
# ---------------------------------------------------------------------------

def load_inception(device):
    """Load pretrained InceptionV3 with a hook to extract pool3 features."""
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.eval()
    inception.to(device)

    # Hook captures the 2048-dim avgpool output (pool3 layer)
    pool3_features = {}
    def hook(module, input, output):
        pool3_features["feats"] = output.squeeze(-1).squeeze(-1)

    inception.avgpool.register_forward_hook(hook)

    return inception, pool3_features


@torch.no_grad()
def get_inception_outputs(images, inception, pool3_features, batch_size=128, device="cuda"):
    """Run images through InceptionV3.

    Returns:
        features: (N, 2048) pool3 features for FID.
        probs:    (N, 1000) softmax class probabilities for IS.
    """
    all_features = []
    all_probs = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)

        # Resize to 299x299 (Inception input size)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)

        logits = inception(batch)
        # inception() returns InceptionOutputs namedtuple during training,
        # plain tensor in eval mode — handle both
        if hasattr(logits, "logits"):
            logits = logits.logits

        probs = F.softmax(logits, dim=-1)
        all_features.append(pool3_features["feats"].cpu())
        all_probs.append(probs.cpu())

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Inception: {min(i + batch_size, len(images))}/{len(images)}")

    return torch.cat(all_features, dim=0).numpy(), torch.cat(all_probs, dim=0).numpy()


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(real_feats, gen_feats):
    """Compute Fréchet Inception Distance between two feature sets.

    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·sqrtm(Σ_r·Σ_g))
    """
    from scipy.linalg import sqrtm

    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_g = np.cov(gen_feats, rowvar=False)

    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)

    # Numerical instability guard
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    return fid


# ---------------------------------------------------------------------------
# IS
# ---------------------------------------------------------------------------

def compute_is(probs, n_splits=10):
    """Compute Inception Score from class probability arrays.

    IS = exp(E_x[KL(p(y|x) || p(y))])

    Args:
        probs: (N, 1000) softmax probabilities.
        n_splits: Number of chunks for mean/std estimation.

    Returns:
        (mean_IS, std_IS)
    """
    scores = []
    chunk_size = len(probs) // n_splits

    for i in range(n_splits):
        chunk = probs[i * chunk_size : (i + 1) * chunk_size]
        p_y = chunk.mean(axis=0, keepdims=True)                   # marginal p(y)
        kl = chunk * (np.log(chunk + 1e-10) - np.log(p_y + 1e-10))
        kl = kl.sum(axis=1)                                        # KL per image
        scores.append(np.exp(kl.mean()))

    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path,
    n_eval=50000,
    batch_size=128,
    data_dir="./data",
    output_dir=".",
    device="cuda",
    image_size=32,
):
    """Compute FID and IS for a trained DDPM model.

    Args:
        checkpoint_path: Path to saved checkpoint (EMA weights used).
        n_eval: Number of samples to generate (50k for paper comparison).
        batch_size: Batch size for both generation and Inception forward passes.
        data_dir: CIFAR-10 data directory.
        output_dir: Where to save eval_results.txt.
        device: Device string.
        image_size: Image spatial size.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}")

    # --- Load generative model ---
    model = UNet(
        in_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        image_size=image_size,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    print(f"Loaded checkpoint from step {checkpoint['step']}")

    diffusion = GaussianDiffusion(T=1000, device=device)

    # --- Generate samples ---
    print(f"\nGenerating {n_eval} samples in batches of {batch_size}...")
    all_samples = []
    generated = 0

    while generated < n_eval:
        this_batch = min(batch_size, n_eval - generated)
        samples = diffusion.p_sample_loop(model, (this_batch, 3, image_size, image_size))
        # Denormalize [-1,1] -> [0,1]
        samples = (samples + 1.0) / 2.0
        samples = samples.clamp(0.0, 1.0).cpu()
        all_samples.append(samples)
        generated += this_batch
        print(f"  Generated {generated}/{n_eval}")

    gen_images = torch.cat(all_samples, dim=0)   # (N, 3, 32, 32) in [0,1]

    # --- Load real CIFAR-10 training images ---
    print("\nLoading real CIFAR-10 training images...")
    real_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=transforms.ToTensor()             # [0,1], no normalization
    )
    real_loader = torch.utils.data.DataLoader(
        real_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    real_images = torch.cat([x for x, _ in real_loader], dim=0)  # (50000, 3, 32, 32)
    print(f"  Loaded {len(real_images)} real images")

    # --- Load Inception ---
    print("\nLoading InceptionV3...")
    inception, pool3_features = load_inception(device)

    # --- Extract features ---
    print("\nExtracting features for generated images...")
    gen_feats, gen_probs = get_inception_outputs(gen_images, inception, pool3_features,
                                                  batch_size=batch_size, device=device)

    print("\nExtracting features for real images...")
    real_feats, _ = get_inception_outputs(real_images, inception, pool3_features,
                                           batch_size=batch_size, device=device)

    # --- Compute metrics ---
    print("\nComputing FID...")
    fid = compute_fid(real_feats, gen_feats)

    print("Computing IS...")
    is_mean, is_std = compute_is(gen_probs)

    # --- Report ---
    print("\n" + "=" * 50)
    print(f"  FID:  {fid:.2f}       (paper: 3.17)")
    print(f"  IS:   {is_mean:.2f} ± {is_std:.2f}  (paper: 9.46 ± 0.11)")
    print("=" * 50)

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "eval_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Step: {checkpoint['step']}\n")
        f.write(f"N_eval: {n_eval}\n\n")
        f.write(f"FID:  {fid:.4f}  (paper: 3.17)\n")
        f.write(f"IS:   {is_mean:.4f} +/- {is_std:.4f}  (paper: 9.46 +/- 0.11)\n")
    print(f"\nResults saved to {results_path}")
