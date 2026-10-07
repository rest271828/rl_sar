#!/usr/bin/env python3
"""Sim smoke: closed-loop EP Student via Isaac Gym env + go2_ep_student sidecar.

rl_sar Gazebo/MuJoCo on this host is unavailable (no ROS, no mujoco build, no docker).
This bridge uses Extreme Parkour's Go2 Isaac Gym sim (the training/eval physics) and
drives it with the SAME base_jit + vision_weight stack as infer_offline.py.

Not a Gazebo drop-in; it proves obs+depth → sidecar → action → env.step.

Example:
  source /home/yihan/extreme-parkour/activate.sh
  cd /home/yihan/extreme-parkour/rest271828-rl_sar
  python policy/go2/go2_ep_student/sim_isaac_bridge.py --steps 200 --num-envs 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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


def main() -> int:
    ap = argparse.ArgumentParser(description="EP Student Isaac Gym closed-loop via rl_sar sidecar")
    ap.add_argument("--ep-root", default=DEFAULT_EP_ROOT)
    ap.add_argument("--base-jit", default=None)
    ap.add_argument("--vision-weight", default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args, unknown = ap.parse_known_args()

    base_path = args.base_jit or _resolve_weight("base_jit.pt", DEFAULT_BASE)
    vision_path = args.vision_weight or _resolve_weight("vision_weight.pt", DEFAULT_VISION)

    # Forwarded into EP get_args / update_cfg_from_args (--num_envs survives camera headless defaults)
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

    import isaacgym  # noqa: F401  — before torch
    import torch
    import legged_gym.envs  # noqa: F401  — registers go2
    from legged_gym.utils import get_args, task_registry
    from ep_depth_encoder import RecurrentDepthBackbone

    ep_args = get_args()
    env_cfg, _train_cfg = task_registry.get_cfgs(name=ep_args.task)
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 60
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    # Prefer flat/step smoke mix; make_env→update_cfg may partially rewrite camera terrain —
    # still useful when it sticks; closed-loop validity does not depend on exact mix.
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False

    print(f"[sim_isaac_bridge] base_jit={base_path}")
    print(f"[sim_isaac_bridge] vision_weight={vision_path}")
    print(f"[sim_isaac_bridge] num_envs={args.num_envs} steps={args.steps} device={args.device}")

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

    rew_acc = []
    act_amax = []
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
            actions = policy(obs_in, latent)
            actions = actions.clamp(-ACTION_CLIP, ACTION_CLIP)
            act_amax.append(float(actions.abs().max().item()))

            obs, _, rews, dones, infos = env.step(actions)
            rew_acc.append(float(rews.mean().item()))
            if dones.any():
                encoder.reset_hidden()
            if "depth" in infos and infos["depth"] is not None:
                depth = infos["depth"].to(device)
            else:
                depth = env.depth_buffer[:, -2].to(device)

            if step % 50 == 0 or step == args.steps - 1:
                print(
                    f"step={step:4d} rew_mean={rew_acc[-1]:+.4f} "
                    f"amax={act_amax[-1]:.3f} "
                    f"vx_cmd={float(env.commands[0, 0]):.2f} "
                    f"vx={float(env.base_lin_vel[0, 0]):.2f} "
                    f"finite={bool(torch.isfinite(actions).all())}"
                )

    dt = time.time() - t0
    print(
        f"OK closed-loop: steps={args.steps} envs={env.num_envs} "
        f"rew_mean={sum(rew_acc)/len(rew_acc):.4f} "
        f"amax_max={max(act_amax):.3f} wall_s={dt:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
