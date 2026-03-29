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

    def __init__(
        self,
        T=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cpu",
        prediction_target="epsilon",
        objective_type="l_simple",
        variance_mode="fixed",
    ):
        self.T = T
        self.device = device
        self.prediction_target = prediction_target
        self.objective_type = objective_type
        self.variance_mode = variance_mode

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

        # True posterior variance used in the variational bound
        self.beta_tilde = ((1.0 - torch.cat([torch.tensor([1.0], device=device), self.alpha_bar[:-1]]))
                   / (1.0 - self.alpha_bar)) * self.beta
        self.beta_tilde = torch.clamp(self.beta_tilde, min=1e-20)

    def _extract(self, arr, t, x):
        return arr[t][:, None, None, None].to(x.device)

    def q_posterior_mean(self, x0, x_t, t):
        """Posterior mean q(x_{t-1} | x_t, x_0)."""
        alpha_t = self._extract(self.alpha, t, x_t)
        alpha_bar_t = self._extract(self.alpha_bar, t, x_t)

        t_prev = torch.clamp(t - 1, min=0)
        alpha_bar_prev = self._extract(self.alpha_bar, t_prev, x_t)

        coeff1 = torch.sqrt(alpha_bar_prev) * (1.0 - alpha_t) / (1.0 - alpha_bar_t)
        coeff2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        return coeff1 * x0 + coeff2 * x_t

    def predict_x0_from_epsilon(self, x_t, t, epsilon):
        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bar, t, x_t)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bar, t, x_t)
        return (x_t - sqrt_one_minus_alpha_bar_t * epsilon) / sqrt_alpha_bar_t

    def _split_model_output(self, model_output, channels):
        if self.variance_mode == "learned":
            return model_output[:, :channels], model_output[:, channels:]
        return model_output, None

    def _sigma2_from_var_pred(self, var_pred):
        return F.softplus(var_pred) + 1e-5

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
        """Compute training loss for selected prediction/objective/variance ablation mode."""
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)

        model_output = model(x_t, t)
        pred, var_pred = self._split_model_output(model_output, x0.shape[1])

        if self.prediction_target == "epsilon":
            pred_epsilon = pred
            pred_x0 = self.predict_x0_from_epsilon(x_t, t, pred_epsilon)
            pred_mu = self.q_posterior_mean(pred_x0, x_t, t)
            target_simple = noise
        elif self.prediction_target == "mu":
            pred_mu = pred
            target_simple = self.q_posterior_mean(x0, x_t, t)
        else:
            raise ValueError(f"Unknown prediction_target: {self.prediction_target}")

        if self.objective_type == "l_simple":
            return F.mse_loss(pred, target_simple)

        if self.objective_type == "mse":
            target_mu = self.q_posterior_mean(x0, x_t, t)
            return F.mse_loss(pred_mu, target_mu)

        if self.objective_type != "l_vlb":
            raise ValueError(f"Unknown objective_type: {self.objective_type}")

        target_mu = self.q_posterior_mean(x0, x_t, t)

        if self.variance_mode == "fixed":
            sigma2_t = self._extract(self.beta, t, x_t)
            return ((target_mu - pred_mu) ** 2 / (2.0 * sigma2_t)).mean()

        if self.variance_mode == "learned":
            sigma2_theta = self._sigma2_from_var_pred(var_pred)
            beta_tilde_t = self._extract(self.beta_tilde, t, x_t)
            kl = 0.5 * (
                (beta_tilde_t / sigma2_theta)
                + ((target_mu - pred_mu) ** 2 / sigma2_theta)
                - 1.0
                + torch.log(sigma2_theta)
                - torch.log(beta_tilde_t)
            )
            return kl.mean()

        raise ValueError(f"Unknown variance_mode: {self.variance_mode}")

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """Single reverse step: sample x_{t-1} given x_t.

        x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta) + sigma_t * z
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        model_output = model(x_t, t_tensor)
        pred, var_pred = self._split_model_output(model_output, x_t.shape[1])

        if self.prediction_target == "epsilon":
            coeff = self.one_minus_alpha_over_sqrt_one_minus_alpha_bar[t]
            mean = (1.0 / self.sqrt_alpha[t]) * (x_t - coeff * pred)
        else:
            pred_x0 = torch.clamp(pred, -1.0, 1.0)
            mean = self.q_posterior_mean(pred_x0, x_t, t_tensor)

        if self.variance_mode == "learned":
            sigma_t = torch.sqrt(self._sigma2_from_var_pred(var_pred))
        else:
            sigma_t = self.sigma[t]

        if t > 0:
            noise = torch.randn_like(x_t)
            return mean + sigma_t * noise
        else:
            return mean

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """Full reverse process: generate images from pure noise.

        Algorithm 2 from the paper.
        """
        x = torch.randn(shape, device=self.device)

        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)

        return x

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
            torch.linspace(self.T - 1, 0, n_frames).tolist()
        )

        x = torch.randn(shape, device=self.device)
        frames = []

        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
            if t in save_at:
                frames.append(x.clone())

        # frames[0] = most noisy saved state, frames[-1] = final image
        return frames
