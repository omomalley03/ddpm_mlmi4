"""
Gaussian diffusion process for DDPM (Ho et al. 2020).

Implements:
- Linear beta schedule
- Forward process q(x_t | x_0) (Algorithm 1)
- Reverse process p(x_{t-1} | x_t) (Algorithm 2)
- Simplified training loss L_simple
"""

import torch
import torch.nn.functional as F


class GaussianDiffusion:
    """DDPM forward and reverse diffusion process.

    Args:
        T: Number of diffusion steps (1000).
        beta_start: Starting beta value (1e-4).
        beta_end: Ending beta value (0.02).
        device: Torch device.
    """

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        # Linear beta schedule
        self.beta = torch.linspace(beta_start, beta_end, T, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Precompute useful quantities
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.one_minus_alpha_over_sqrt_one_minus_alpha_bar = (
            (1.0 - self.alpha) / self.sqrt_one_minus_alpha_bar
        )

        # Sampling variance: sigma_t^2 = beta_t (fixed, not learned)
        self.sigma = torch.sqrt(self.beta)

    def q_sample(self, x0, t, noise=None):
        """Forward process: sample x_t given x_0.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    def p_losses(self, model, x0, t):
        """Compute L_simple = ||epsilon - epsilon_theta(x_t, t)||^2.

        Algorithm 1 from the paper.
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)
        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)
        return loss


    # BELOW FUNCTIONS ARE FOR SAMPLING (REVERSE PROCESS) ONLY - NOT USED DURING TRAINING
    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """Single reverse step: sample x_{t-1} given x_t.
        Equation 11 in Ho et al
        x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta) + sigma_t * z
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        predicted_noise = model(x_t, t_tensor)

        # Compute mean
        coeff = self.one_minus_alpha_over_sqrt_one_minus_alpha_bar[t]
        mean = (1.0 / self.sqrt_alpha[t]) * (x_t - coeff * predicted_noise)

        if t > 0:
            noise = torch.randn_like(x_t)
            return mean + self.sigma[t] * noise
        else:
            return mean

    @torch.no_grad()
    def p_sample_loop(self, model, shape, noise=None):
        """Full reverse process: generate images from pure noise.

        Algorithm 2 from the paper.
        """
        x = noise if noise is not None else torch.randn(shape, device=self.device)

        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)

        return x


    def slerp(self, z1: torch.Tensor, z2: torch.Tensor, alpha: float) -> torch.Tensor:
        # Flatten to vectors for the dot product, then reshape back
        z1_flat = z1.view(z1.shape[0], -1)
        z2_flat = z2.view(z2.shape[0], -1)

        # Normalize
        z1_norm = F.normalize(z1_flat, dim=-1)
        z2_norm = F.normalize(z2_flat, dim=-1)

        # Angle between them (clamp for numerical safety)
        dot = (z1_norm * z2_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
        omega = torch.acos(dot)          # shape: (B, 1)

        # Slerp formula; fall back to lerp if omega ≈ 0
        sin_omega = torch.sin(omega)
        safe = sin_omega.abs() > 1e-6
        coeff1 = torch.where(safe, torch.sin((1 - alpha) * omega) / sin_omega, torch.tensor(1 - alpha))
        coeff2 = torch.where(safe, torch.sin(alpha * omega) / sin_omega, torch.tensor(alpha))

        interp_flat = coeff1 * z1_flat + coeff2 * z2_flat
        return interp_flat.view_as(z1)


    @torch.no_grad()
    def samples_interpolate(self, model, shape, interp_steps=10):
        noise1 = torch.randn(shape, device=self.device)
        noise2 = torch.randn(shape, device=self.device)

        frames = []
        for i in range(interp_steps):
            alpha = i / (interp_steps - 1)
            xT = self.slerp(noise1, noise2, alpha)          # <-- slerp, not lerp
            frames.append(self.p_sample_loop(model, shape, xT))

        return frames
    


    @torch.no_grad()
    def p_sample_loop_progressive(self, model, shape, n_frames=10):
        """Reverse process capturing intermediate denoising states.

        Returns a list of n_frames tensors evenly spaced across the
        denoising trajectory (t=T-1 down to t=0), suitable for
        visualising the noise-to-image progression.
        """
        # Timesteps at which to save a frame, spread across [0, T-1]
        # Always include t=0 (final image) and t=T-1 (pure noise)
        save_at = set(
            round(t) for t in
            torch.linspace((self.T)  - 1, 0, n_frames).tolist()
        )

        x = torch.randn(shape, device=self.device)
        frames = []

        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
            if t in save_at:
                frames.append(x.clone())

        # frames[0] = most noisy saved state, frames[-1] = final image
        return frames
