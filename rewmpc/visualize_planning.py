"""
Visualize raw flow prior proposals vs gradient-refined proposals.

For each planning step: renders the scene with both trajectory sets overlaid as
EE-position paths projected onto the camera image.
  Blue  = raw flow prior proposals (before gradient refinement)
  Orange = gradient-refined proposals (selected mode = bright orange, thicker)

Usage:
    python visualize_planning.py checkpoint=/path/to/final.pt \
        task=mw-button-press-topdown seed=1 exp_name=single_task \
        viz_episodes=2 viz_camera=corner2 flow_warm_noise=0.0
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import mujoco
import imageio
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

import hydra
from omegaconf import OmegaConf
from common.parser import parse_cfg
from common.seed import set_seed
from agent import ReWMPC
from envs import make_env
from envs.metaworld import MetaWorldWrapper, Timeout, _get_mt10, _task_id


# ──────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────

def make_viz_env(cfg, camera_name='corner2'):
    """Make env with a specific named camera for consistent projection."""
    task_id = _task_id(cfg.task)
    mt10 = _get_mt10(cfg.seed)
    EnvCls = mt10.train_classes[task_id]
    raw_env = EnvCls(render_mode='rgb_array', camera_name=camera_name)
    tasks = [t for t in mt10.train_tasks if t.env_name == task_id]
    env = MetaWorldWrapper(raw_env, tasks, cfg)
    env = Timeout(env, max_episode_steps=100)
    return env


def save_restore_state(env):
    """Return a closure that restores the full raw env state (MuJoCo + MetaWorld counters)."""
    raw = env.unwrapped
    qpos          = raw.data.qpos.copy()
    qvel          = raw.data.qvel.copy()
    time          = float(raw.data.time)
    path_length   = int(raw.curr_path_length)

    def restore():
        raw.data.qpos[:]      = qpos
        raw.data.qvel[:]      = qvel
        raw.data.time         = time
        raw.curr_path_length  = path_length
        mujoco.mj_forward(raw.model, raw.data)

    return restore


def simulate_ee_trajectories(env, tau_np):
    """
    Simulate M trajectories from the current env state (non-destructive).
    Calls raw env step 2x per action (matching MetaWorldWrapper frame skip).

    Args:
        env:     wrapped env (Timeout → MetaWorldWrapper → raw)
        tau_np:  (M, H, A) numpy actions in [-1, 1]

    Returns:
        ee_xyz:  (M, H+1, 3) end-effector world positions
    """
    raw = env.unwrapped
    ee_id = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_SITE, 'endEffector')
    restore = save_restore_state(env)

    M, H, A = tau_np.shape
    ee_xyz = np.zeros((M, H + 1, 3), dtype=np.float32)

    for m in range(M):
        restore()
        mujoco.mj_forward(raw.model, raw.data)
        ee_xyz[m, 0] = raw.data.site_xpos[ee_id].copy()
        for h in range(H):
            action = tau_np[m, h]
            for _ in range(2):  # frame skip = 2
                raw.step(action)
            ee_xyz[m, h + 1] = raw.data.site_xpos[ee_id].copy()

    restore()
    return ee_xyz


# ──────────────────────────────────────────────
# Projection
# ──────────────────────────────────────────────

def project_3d_to_2d(xyz_world, model, data, cam_id, W, H):
    """
    Project (N, 3) world-frame points to (N, 2) image pixels using a named camera.
    Points behind the camera are masked with (-1, -1).
    """
    cam_pos = data.cam_xpos[cam_id]                  # (3,)
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)    # columns = camera axes in world frame

    fovy_rad = np.radians(model.cam_fovy[cam_id])
    f = 0.5 * H / np.tan(fovy_rad / 2)
    cx, cy = W / 2.0, H / 2.0

    # World → camera frame
    pts = (cam_mat.T @ (xyz_world.T - cam_pos[:, None])).T  # (N, 3)

    # MuJoCo camera z-axis points BACKWARD; depth = -z for in-front points
    depth = -pts[:, 2]
    valid = depth > 0.01
    safe_depth = np.where(valid, depth, 1.0)
    u = np.where(valid, f * pts[:, 0] / safe_depth + cx, -1.0)
    v = np.where(valid, -f * pts[:, 1] / safe_depth + cy, -1.0)

    uv = np.stack([u, v], axis=-1)
    uv[~valid] = -1.0
    return uv, valid


# ──────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────

def annotate_frame(frame, raw_uv, refined_uv, best_idx):
    """
    Overlay raw (blue) and refined (orange) EE trajectory proposals on frame.

    Args:
        frame:      (H, W, 3) uint8 numpy array
        raw_uv:     (M, T, 2) pixel coords for raw proposals
        refined_uv: (M, T, 2) pixel coords for refined proposals
        best_idx:   index of selected (lowest J) mode

    Returns:
        annotated frame as (H, W, 3) uint8 numpy array
    """
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    W, H = img.size
    M, T, _ = raw_uv.shape

    def valid_pt(uv):
        return 0 <= uv[0] < W and 0 <= uv[1] < H

    def draw_traj(uv_traj, color, endpoint_r, line_w):
        pts = [(float(uv_traj[t, 0]), float(uv_traj[t, 1])) for t in range(T)]
        valid = [valid_pt(p) for p in pts]
        # lines between consecutive valid waypoints
        for t in range(T - 1):
            if valid[t] and valid[t + 1]:
                draw.line([pts[t], pts[t + 1]], fill=color, width=line_w)
        # endpoint dot only (last valid point)
        for t in range(T - 1, -1, -1):
            if valid[t]:
                p = pts[t]
                r = endpoint_r
                draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)
                break

    # Raw proposals — bright blue, very thin, drawn first (below)
    for m in range(M):
        draw_traj(raw_uv[m], color=(130, 180, 255, 200), endpoint_r=1, line_w=1)

    # Refined non-selected — orange, thin
    for m in range(M):
        if m != best_idx:
            draw_traj(refined_uv[m], color=(255, 165, 0, 160), endpoint_r=1, line_w=1)

    # Refined selected — bright orange, slightly thicker, drawn on top
    draw_traj(refined_uv[best_idx], color=(255, 200, 0, 255), endpoint_r=2, line_w=2)

    out = Image.alpha_composite(img, overlay).convert('RGB').rotate(180)
    return np.array(out)


# ──────────────────────────────────────────────
# Planning helpers (inline — no agent._plan side effects)
# ──────────────────────────────────────────────

def get_raw_proposals(agent, z_t, h_t, task, cfg):
    """Sample M raw flow proposals. Returns (M, H, A) tensor (detached)."""
    with torch.no_grad():
        context = agent.model.context(h_t, z_t, task)
        tau_raw = agent.flow_prior.sample(context, M=cfg.flow_modes, nfe=cfg.flow_nfe)
    return tau_raw.detach()


def refine_proposals(agent, tau_init, z_t, h_t, task, cfg):
    """
    Run K gradient steps on tau_init and return refined tau + best index.
    Returns (M, H, A) tensor (detached) and best_idx (int).
    """
    M = tau_init.shape[0]
    z_exp = z_t.expand(M, -1)
    h_exp = h_t.expand(M, -1)
    discount = float(agent.discount[task].item() if cfg.multitask else agent.discount)

    frozen = (list(agent.model.parameters()) +
              list(agent.reward_surrogate.parameters()) +
              list(agent.value_surrogate.parameters()))
    saved = [p.requires_grad for p in frozen]
    for p in frozen:
        p.requires_grad_(False)

    try:
        with torch.enable_grad():
            tau = tau_init.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([tau], lr=cfg.grad_plan_lr)
            for _ in range(cfg.grad_plan_steps):
                opt.zero_grad()
                J = agent._planning_cost(tau, z_exp, h_exp, tau_init, task, discount)
                J.sum().backward()
                nn.utils.clip_grad_norm_([tau], cfg.grad_clip_norm)
                opt.step()
                with torch.no_grad():
                    tau.data.clamp_(-1., 1.)
        with torch.no_grad():
            final_J = agent._planning_cost(
                tau.detach(), z_exp, h_exp, tau_init, task, discount)
    finally:
        for p, req in zip(frozen, saved):
            p.requires_grad_(req)

    best_idx = int(final_J.argmin().item())
    return tau.detach(), best_idx


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

@hydra.main(config_name='config', config_path='.')
def main(cfg):
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    # Extra viz-specific flags (with defaults so they're optional)
    viz_episodes = int(getattr(cfg, 'viz_episodes', 1))
    viz_camera   = str(getattr(cfg, 'viz_camera', 'corner2'))
    viz_every    = int(getattr(cfg, 'viz_every', 1))    # annotate every N steps
    viz_speed    = float(getattr(cfg, 'viz_speed', 0.3))  # playback speed multiplier

    assert cfg.checkpoint not in (None, '???'), 'Pass checkpoint=/path/to/final.pt'

    # Populate cfg.obs_shape, cfg.action_dim, etc. (make_env does this as a side effect)
    make_env(cfg)

    agent = ReWMPC(cfg)
    agent.load(cfg.checkpoint)
    agent.model.eval()
    agent.flow_prior.eval()
    agent.reward_surrogate.eval()
    agent.value_surrogate.eval()
    print(f'Loaded checkpoint: {cfg.checkpoint}')

    env = make_viz_env(cfg, camera_name=viz_camera)
    raw = env.unwrapped
    cam_id = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_CAMERA, viz_camera)
    assert cam_id >= 0, f'Camera {viz_camera!r} not found'

    video_dir = Path(cfg.work_dir) / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(viz_episodes):
        frames = []
        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        h_t = torch.zeros(1, cfg.latent_dim, device=agent.device)
        done = False
        step = 0
        total_r, success = 0.0, 0.0

        while not done:
            with torch.no_grad():
                z_t = agent.model.encode(obs_t, None)

            tau_raw = get_raw_proposals(agent, z_t, h_t, None, cfg)
            tau_ref, best_idx = refine_proposals(agent, tau_raw, z_t, h_t, None, cfg)

            # Render current frame
            mujoco.mj_forward(raw.model, raw.data)
            base_frame = env.render()          # (H_img, W_img, 3)
            H_img, W_img = base_frame.shape[:2]

            if step % viz_every == 0:
                # Simulate trajectories → EE positions
                tau_raw_np = tau_raw.cpu().numpy()
                tau_ref_np = tau_ref.cpu().numpy()

                ee_raw = simulate_ee_trajectories(env, tau_raw_np)   # (M, T, 3)
                ee_ref = simulate_ee_trajectories(env, tau_ref_np)   # (M, T, 3)

                mujoco.mj_forward(raw.model, raw.data)  # restore for projection

                M, T, _ = ee_raw.shape
                raw_uv  = np.zeros((M, T, 2), dtype=np.float32)
                ref_uv  = np.zeros((M, T, 2), dtype=np.float32)

                for m in range(M):
                    raw_uv[m], _  = project_3d_to_2d(ee_raw[m], raw.model, raw.data,
                                                      cam_id, W_img, H_img)
                    ref_uv[m], _  = project_3d_to_2d(ee_ref[m], raw.model, raw.data,
                                                      cam_id, W_img, H_img)

                annotated = annotate_frame(base_frame, raw_uv, ref_uv, best_idx)
            else:
                annotated = base_frame[::-1, ::-1]

            frames.append(annotated)

            # Execute best refined action
            action = tau_ref[best_idx, 0].cpu().numpy()
            obs, reward, done, info = env.step(action)
            total_r += reward
            success = max(success, float(info.get('success', 0)))

            obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                h_t = agent.model.gru_step(z_t, tau_ref[best_idx:best_idx+1, 0], h_t)
            step += 1
            if info.get('success', 0):
                # Linger 5 more steps so the viewer can see the goal state
                for _ in range(5):
                    frames.append(annotated)
                break

        out_path = video_dir / f'plan_viz_ep{ep}.mp4'
        imageio.mimsave(str(out_path), frames, fps=max(1, round(15 * viz_speed)))
        print(f'  ep{ep}  R={total_r:.1f}  success={success:.0f}  '
              f'steps={step}  → {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()
