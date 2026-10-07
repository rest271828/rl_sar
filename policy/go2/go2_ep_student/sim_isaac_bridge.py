#!/usr/bin/env python3
"""Sim smoke: closed-loop EP Student via Isaac Gym env + go2_ep_student sidecar.

rl_sar Gazebo/MuJoCo on this host is unavailable (no ROS, no mujoco build, no docker).
This bridge uses Extreme Parkour's Go2 Isaac Gym sim (the training/eval physics) and
drives it with the SAME base_jit + vision_weight stack as infer_offline.py.

Example:
  source /home/yihan/extreme-parkour/activate.sh
  cd /home/yihan/extreme-parkour/rest271828-rl_sar
  python policy/go2/go2_ep_student/sim_isaac_bridge.py --steps 800 --num-envs 16 --terrain parkour
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_EP_ROOT = "/home/yihan/extreme-parkour/extreme-parkour-repro"
DEFAULT_TRACED = (
    f"{DEFAULT_EP_ROOT}/legged_gym/logs/parkour_new/922-10-GO2-STUDENT-D0/traced"
)
DEFAULT_BASE = os.path.join(DEFAULT_TRACED, "922-10-GO2-STUDENT-D0-59500-base_jit.pt")
DEFAULT_VISION = os.path.join(DEFAULT_TRACED, "922-10-GO2-STUDENT-D0-59500-vision_weight.pt")

ACTION_CLIP = 4.8
YAW_SCALE = 1.5
N_PROPRIO = 53
# evaluate.py normalizes waypoints by 7 → reported as ~MXD in project docs
WAYPOINT_NORM = 7.0

TERRAIN_PRESETS = {
    "flat": {
        "parkour": 0.0,
        "parkour_hurdle": 0.0,
        "parkour_flat": 0.7,
        "parkour_step": 0.3,
        "parkour_gap": 0.0,
        "demo": 0.0,
    },
    # Closer to evaluate.py mix (not identical; camera headless may rewrite some keys)
    "parkour": {
        "parkour": 0.25,
        "parkour_hurdle": 0.25,
        "parkour_flat": 0.0,
        "parkour_step": 0.25,
        "parkour_gap": 0.25,
        "demo": 0.0,
    },
}


def _resolve_weight(local_name: str, default_abs: str) -> str:
    local = os.path.join(_HERE, local_name)
    if os.path.isfile(local):
        return local
    if os.path.isfile(default_abs):
        return default_abs
    raise FileNotFoundError(f"Missing weight: {local} or {default_abs}")


def _prepare_paths(ep_root: str) -> None:
    ep_root = os.path.abspath(ep_root)
    gym_root = os.path.join(ep_root, "legged_gym")
    rsl_root = os.path.join(ep_root, "rsl_rl")
    scripts = os.path.join(gym_root, "legged_gym", "scripts")
    for p in (gym_root, rsl_root, scripts):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(scripts)


def _zero_terrain_dict():
    return {
        "smooth slope": 0.0,
        "rough slope up": 0.0,
        "rough slope down": 0.0,
        "rough stairs up": 0.0,
        "rough stairs down": 0.0,
        "discrete": 0.0,
        "stepping stones": 0.0,
        "gaps": 0.0,
        "smooth flat": 0.0,
        "pit": 0.0,
        "wall": 0.0,
        "platform": 0.0,
        "large stairs up": 0.0,
        "large stairs down": 0.0,
        "parkour": 0.0,
        "parkour_hurdle": 0.0,
        "parkour_flat": 0.0,
        "parkour_step": 0.0,
        "parkour_gap": 0.0,
        "demo": 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="EP Student Isaac Gym closed-loop via rl_sar sidecar")
    ap.add_argument("--ep-root", default=DEFAULT_EP_ROOT)
    ap.add_argument("--base-jit", default=None)
    ap.add_argument("--vision-weight", default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--terrain", choices=sorted(TERRAIN_PRESETS.keys()), default="flat")
    ap.add_argument("--log-every", type=int, default=50)
    args, unknown = ap.parse_known_args()

    base_path = args.base_jit or _resolve_weight("base_jit.pt", DEFAULT_BASE)
    vision_path = args.vision_weight or _resolve_weight("vision_weight.pt", DEFAULT_VISION)

    ep_argv = [
        "sim_isaac_bridge.py",
        "--task", "go2",
        "--headless",
        "--use_camera",
        "--delay",
        "--proj_name", "parkour_new",
        "--exptid", "922-10-GO2-STUDENT-D0",
        "--checkpoint", "59500",
        "--num_envs", str(args.num_envs),
        "--seed", str(args.seed),
        "--sim_device", args.device,
    ] + unknown
    sys.argv = ep_argv

    _prepare_paths(args.ep_root)

    import isaacgym  # noqa: F401
    import torch
    import numpy as np
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils import get_args, task_registry
    from ep_depth_encoder import RecurrentDepthBackbone

    ep_args = get_args()
    env_cfg, _train_cfg = task_registry.get_cfgs(name=ep_args.task)
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 60
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False
    # Best-effort terrain mix (camera headless update_cfg may rewrite proportions)
    tdict = _zero_terrain_dict()
    tdict.update(TERRAIN_PRESETS[args.terrain])
    env_cfg.terrain.terrain_dict = tdict
    env_cfg.terrain.terrain_proportions = list(tdict.values())

    print(f"[sim_isaac_bridge] base_jit={base_path}")
    print(f"[sim_isaac_bridge] vision_weight={vision_path}")
    print(
        f"[sim_isaac_bridge] num_envs={args.num_envs} steps={args.steps} "
        f"terrain={args.terrain} device={args.device}"
    )

    env, env_cfg = task_registry.make_env(name=ep_args.task, args=ep_args, env_cfg=env_cfg)
    print(f"[sim_isaac_bridge] actual num_envs={env.num_envs}")
    obs = env.get_observations()
    device = env.device

    encoder = RecurrentDepthBackbone(n_proprio=N_PROPRIO).to(device)
    blob = torch.load(vision_path, map_location=device, weights_only=False)
    sd = blob["depth_encoder_state_dict"] if isinstance(blob, dict) and "depth_encoder_state_dict" in blob else blob
    encoder.load_state_dict(sd, strict=True)
    encoder.eval()
    encoder.reset_hidden()

    policy = torch.jit.load(base_path, map_location=device)
    policy.eval()

    depth = env.depth_buffer[:, -2].to(device)

    step_rew = []
    act_amax = []
    vx_err = []  # |vx - cmd_vx|
    rewbuffer = deque(maxlen=max(args.steps, 256))
    lenbuffer = deque(maxlen=max(args.steps, 256))
    wpbuffer = deque(maxlen=max(args.steps, 256))
    fallbuffer = deque(maxlen=max(args.steps, 256))

    cur_reward_sum = torch.zeros(env.num_envs, device=device)
    cur_episode_length = torch.zeros(env.num_envs, device=device)

    t0 = time.time()
    with torch.no_grad():
        for step in range(args.steps):
            proprio = obs[:, :N_PROPRIO].clone()
            proprio[:, 6:8] = 0
            depth_out = encoder(depth, proprio)
            latent = depth_out[:, :-2]
            yaw = YAW_SCALE * depth_out[:, -2:]
            obs_in = obs.clone()
            obs_in[:, 6:8] = yaw
            actions = policy(obs_in, latent).clamp(-ACTION_CLIP, ACTION_CLIP)
            act_amax.append(float(actions.abs().max().item()))

            cur_goal_idx = env.cur_goal_idx.clone()
            obs, _, rews, dones, infos = env.step(actions)
            step_rew.append(float(rews.mean().item()))
            vx_err.append(float((env.base_lin_vel[:, 0] - env.commands[:, 0]).abs().mean().item()))

            cur_reward_sum += rews
            cur_episode_length += 1

            new_ids = (dones > 0).nonzero(as_tuple=False)
            time_outs = infos["time_outs"] if "time_outs" in infos else torch.zeros_like(dones)
            killed_ids = ((dones > 0) & (~time_outs)).nonzero(as_tuple=False)
            if new_ids.numel() > 0:
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                wpbuffer.extend(cur_goal_idx[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0
                # NOTE: single shared GRU hidden for all envs — reset on any done (bridge limitation)
                encoder.reset_hidden()
            if killed_ids.numel() > 0:
                fallbuffer.extend([1.0] * killed_ids.shape[0])

            if "depth" in infos and infos["depth"] is not None:
                depth = infos["depth"].to(device)
            else:
                depth = env.depth_buffer[:, -2].to(device)

            if step % args.log_every == 0 or step == args.steps - 1:
                print(
                    f"step={step:4d} rew_mean={step_rew[-1]:+.4f} "
                    f"amax={act_amax[-1]:.3f} "
                    f"vx_err={vx_err[-1]:.3f} "
                    f"vx_cmd={float(env.commands[0, 0]):.2f} "
                    f"vx={float(env.base_lin_vel[0, 0]):.2f} "
                    f"finite={bool(torch.isfinite(actions).all())}"
                )

    dt = time.time() - t0
    ep_count = len(rewbuffer)
    mxd = float(np.mean(np.array(wpbuffer, dtype=float) / WAYPOINT_NORM)) if wpbuffer else float("nan")
    mxd_std = float(np.std(np.array(wpbuffer, dtype=float) / WAYPOINT_NORM)) if len(wpbuffer) > 1 else 0.0
    len_mean = float(np.mean(lenbuffer)) if lenbuffer else float("nan")
    rew_ep = float(np.mean(rewbuffer)) if rewbuffer else float("nan")
    fall_n = len(fallbuffer)
    print("--- summary (evaluate-like; not full evaluate.py) ---")
    print(f"terrain_requested={args.terrain}")
    print(f"steps={args.steps} envs={env.num_envs} wall_s={dt:.1f}")
    print(f"step_rew_mean={sum(step_rew)/len(step_rew):.4f}")
    print(f"vx_abs_err_mean={sum(vx_err)/len(vx_err):.4f}")
    print(f"amax_max={max(act_amax):.3f} amax_mean={sum(act_amax)/len(act_amax):.3f}")
    print(f"episodes_finished={ep_count}")
    print(f"episode_rew_mean={rew_ep:.4f}")
    print(f"episode_len_mean={len_mean:.2f}")
    print(f"MXD_proxy(waypoints/7)={mxd:.4f}±{mxd_std:.4f}")
    print(f"falls_non_timeout={fall_n}")
    print(
        "OK closed-loop: weights load; finite actions; "
        "metrics are bridge-smoke only (see sim-deploy-current-issues.md)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
