"""
UNet-based Diffusion Prior over action trajectories.

Drop-in replacement for FlowPrior. Uses the same 1D conditional UNet
architecture from the official Diffusion Policy paper (Chi et al. 2023),
adapted for our short horizon (H=5) with down_dims=[256, 512] (2 levels).

Trains with DDPM (epsilon prediction) and samples with DDIM.
Supports optional context conditioning and return-weighted training.

Interface matches FlowPrior exactly:
  sample(context, M, nfe)        — nfe = DDIM denoising steps
  cfm_loss(tau_1, context, weights)
  log_prior_penalty(tau, tau_init)

Key design choices vs. the MLP prior:
  - 1D UNet with FiLM conditioning: models temporal structure in H=5 horizon
  - down_dims=[256, 512]: 2 down-levels to avoid over-squashing H=5 → 3 → identity
  - EMA: on by default (critical for stable DDPM inference)
  - Optimizer: AdamW with betas=[0.95, 0.999] (matches official DP)
  - LR schedule: cosine with warmup (handled in offline_trainer.py)
"""

import copy
import math
import logging

import torch
import torch.nn as nn
import einops
from diffusers import DDIMScheduler, DDPMScheduler

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Building blocks (from official Diffusion Policy)
# ──────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=t.device) * -freq)
        emb = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish."""
    def __init__(self, in_ch, out_ch, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning on diffusion timestep + global context."""
    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=3, n_groups=8,
                 cond_predict_scale=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_ch, out_ch, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_ch, out_ch, kernel_size, n_groups=n_groups),
        ])
        cond_out = out_ch * 2 if cond_predict_scale else out_ch
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_ch
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_out),
            nn.Unflatten(-1, (-1, 1)),  # (B, cond_out, 1)
        )
        self.residual_conv = (nn.Conv1d(in_ch, out_ch, 1)
                              if in_ch != out_ch else nn.Identity())

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)          # (B, cond_out, 1)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            out = embed[:, 0] * out + embed[:, 1]
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """
    1D UNet noise predictor for diffusion policy.

    Args:
        input_dim:               action_dim (trajectory channel dim)
        global_cond_dim:         conditioning vector dim (context_dim or 0)
        diffusion_step_embed_dim: timestep embedding size
        down_dims:               channel sizes per UNet level
        kernel_size:             conv kernel size
        n_groups:                GroupNorm groups
        cond_predict_scale:      FiLM scale+bias vs bias-only
    """
    def __init__(self, input_dim, global_cond_dim=0,
                 diffusion_step_embed_dim=256, down_dims=(256, 512),
                 kernel_size=5, n_groups=8, cond_predict_scale=True):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Middle
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
        ])

        # Down
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        # Up
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info("ConditionalUnet1D: %d parameters", n_params)

    def forward(self, sample, timestep, global_cond=None):
        """
        sample:      (B, H, action_dim)  — noisy trajectory
        timestep:    (B,) long or scalar
        global_cond: (B, global_cond_dim) or None

        Returns: (B, H, action_dim) predicted noise
        """
        B, H_orig, _ = sample.shape
        # Pad to even length so down/upsample is symmetric (e.g. H=5 → 6 → 3 → 6 → crop to 5)
        if H_orig % 2 != 0:
            sample = torch.nn.functional.pad(sample, (0, 0, 0, 1))

        # (B, H, C) → (B, C, H) for Conv1d
        x = einops.rearrange(sample, 'b h c -> b c h')

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)  # (B, dsed)
        if global_cond is not None and global_cond.shape[-1] > 0:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat([x, h.pop()], dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        out = einops.rearrange(x, 'b c h -> b h c')
        return out[:, :H_orig]  # crop back to original horizon length


# ──────────────────────────────────────────────
# DiffusionPrior
# ──────────────────────────────────────────────

class DiffusionPrior(nn.Module):
    """
    UNet-based diffusion prior π_θ(τ | c_t).

    Config keys (all optional, with defaults):
      dp_down_dims:              tuple  = (256, 512)
      dp_diffusion_step_embed:   int    = 256
      dp_kernel_size:            int    = 5
      dp_n_groups:               int    = 8
      dp_num_train_timesteps:    int    = 100
      dp_num_infer_timesteps:    int    = 100
      dp_use_ema:                bool   = True
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._T = getattr(cfg, 'dp_num_train_timesteps', 100)

        down_dims   = tuple(getattr(cfg, 'dp_down_dims', (256, 512)))
        dsed        = getattr(cfg, 'dp_diffusion_step_embed', 256)
        kernel_size = getattr(cfg, 'dp_kernel_size', 5)
        n_groups    = getattr(cfg, 'dp_n_groups', 8)

        self._net = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=cfg.flow_context_dim,
            diffusion_step_embed_dim=dsed,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=True,
        )

        # EMA shadow copy (on by default for diffusion)
        self._use_ema = getattr(cfg, 'dp_use_ema', True)
        self._ema_decay = 0.9999
        if self._use_ema:
            self._ema_net = copy.deepcopy(self._net)
            for p in self._ema_net.parameters():
                p.requires_grad_(False)
        else:
            self._ema_net = None

        # Schedulers
        self._train_scheduler = DDPMScheduler(
            num_train_timesteps=self._T,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )
        self._infer_scheduler = DDIMScheduler(
            num_train_timesteps=self._T,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon',
        )

    def cfm_loss(self, tau_1, context, weights=None):
        """
        DDPM training loss (epsilon prediction).
        Named cfm_loss for drop-in compatibility with FlowPrior.

        tau_1:   (B, H*A) flattened demonstration trajectories
        context: (B, context_dim)  — may be (B, 0) for unconditional
        weights: (B,) optional return-based importance weights
        """
        B = tau_1.shape[0]
        # Reshape to (B, H, A) for UNet
        tau_1_2d = tau_1.view(B, self.cfg.horizon, self.cfg.action_dim)

        t_idx = torch.randint(0, self._T, (B,), device=tau_1.device)
        noise = torch.randn_like(tau_1_2d)

        tau_t = self._train_scheduler.add_noise(tau_1_2d, noise, t_idx)
        noise_pred = self._net(tau_t, t_idx, global_cond=context if context.shape[-1] > 0 else None)

        loss = (noise_pred - noise).pow(2).sum(dim=(-2, -1))  # (B,)
        if weights is not None:
            return (weights * loss).mean()
        return loss.mean()

    @torch.no_grad()
    def sample(self, context, M, nfe=None, tau_warm=None, s_start=0.0):
        """
        Generate M trajectories via DDIM denoising.

        context:  (1, context_dim) or (M, context_dim)
        M:        number of parallel proposals
        nfe:      DDIM steps (default: dp_num_infer_timesteps)
        tau_warm / s_start: accepted for interface compatibility, unused.
        """
        if nfe is None:
            nfe = getattr(self.cfg, 'dp_num_infer_timesteps', 100)
        if context.shape[0] == 1:
            context = context.expand(M, -1)

        net = self._ema_net if (self._use_ema and self._ema_net is not None) else self._net
        global_cond = context if context.shape[-1] > 0 else None

        self._infer_scheduler.set_timesteps(nfe)

        tau = torch.randn(M, self.cfg.horizon, self.cfg.action_dim, device=context.device)
        for t in self._infer_scheduler.timesteps:
            noise_pred = net(tau, t, global_cond=global_cond)
            tau = self._infer_scheduler.step(noise_pred, t, tau).prev_sample

        tau = tau.clamp(-1., 1.)
        return tau  # (M, H, A) — already shaped correctly

    def log_prior_penalty(self, tau, tau_init):
        """Conservatism penalty — identical to FlowPrior."""
        return (tau - tau_init).pow(2).sum(dim=(-2, -1))

    def update_ema(self):
        if self._ema_net is None:
            return
        with torch.no_grad():
            for ema_p, p in zip(self._ema_net.parameters(), self._net.parameters()):
                ema_p.data.mul_(self._ema_decay).add_(p.data, alpha=1.0 - self._ema_decay)
