#!/usr/bin/env python3
"""Sim smoke / MXD measure: EP Student via Isaac Gym + go2_ep_student sidecar.

Backends:
  jit  — deploy path: vision_weight + base_jit (estimator inside JIT)
  ckpt — evaluate.py path: depth_encoder + depth_actor from model_*.pt

Example:
  source /home/yihan/extreme-parkour/activate.sh
  cd /home/yihan/extreme-parkour/rest271828-rl_sar
  python policy/go2/go2_ep_student/sim_isaac_bridge.py --backend jit --steps 1500 --num-envs 64 --terrain parkour
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
DEFAULT_CKPT_DIR = (
    f"{DEFAULT_EP_ROOT}/legged_gym/logs/parkour_new/922-10-GO2-STUDENT-D0"
)

ACTION_CLIP = 4.8  # = clip_actions(1.2) / action_scale(0.25); env also clips
YAW_SCALE = 1.5
N_PROPRIO = 53
WAYPOINT_NORM = 7.0

# evaluate.py terrain_dict (headless+camera will still rewrite some keys in update_cfg)
EVAL_TERRAIN = {
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
    "parkour": 0.25,
    "parkour_hurdle": 0.25,
    "parkour_flat": 0.0,
    "parkour_step": 0.25,
    "parkour_gap": 0.25,
    "demo": 0.0,
}

FLAT_TERRAIN = dict(EVAL_TERRAIN)
FLAT_TERRAIN.update({
    "parkour": 0.0,
    "parkour_hurdle": 0.0,
    "parkour_flat": 0.7,
    "parkour_step": 0.3,
    "parkour_gap": 0.0,
    "demo": 0.0,
})


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


def _apply_evaluate_env_cfg(env_cfg, ep_args, terrain_name: str, num_envs: int):
    """Match evaluate.py overrides as closely as practical."""
    env_cfg.env.num_envs = num_envs
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    tdict = FLAT_TERRAIN if terrain_name == "flat" else EVAL_TERRAIN
    env_cfg.terrain.terrain_dict = dict(tdict)
    env_cfg.terrain.terrain_proportions = list(tdict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False
    # Go2 keeps trained pitch; only A1 forces [0,1] in evaluate.py
    if getattr(ep_args, "task", "go2") == "a1":
        env_cfg.depth.angle = [0, 1]
    # else leave Go2ParkourCfg.depth.angle = [21.2, 24.6]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    return env_cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="EP Student Isaac Gym bridge / MXD measure")
    ap.add_argument("--ep-root", default=DEFAULT_EP_ROOT)
    ap.add_argument("--base-jit", default=None)
    ap.add_argument("--vision-weight", default=None)
    ap.add_argument("--steps", type=int, default=1500, help="evaluate.py uses 1500")
    ap.add_argument("--num-envs", type=int, default=192, help="evaluate headless camera default")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--terrain", choices=["flat", "parkour"], default="parkour")
    ap.add_argument("--backend", choices=["jit", "ckpt"], default="jit",
                    help="jit=deploy TorchScript; ckpt=evaluate.py depth_actor path")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--preclamp", action="store_true",
                    help="clamp actions to +/-4.8 before env.step (env already clips)")
    ap.add_argument("--reset-gru-on-done", action="store_true",
                    help="zero per-env GRU hidden on episode done (deploy-correct; evaluate.py does NOT)")
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
        "--no_wandb",
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
    env_cfg, train_cfg = task_registry.get_cfgs(name=ep_args.task)
    env_cfg = _apply_evaluate_env_cfg(env_cfg, ep_args, args.terrain, args.num_envs)

    print(f"[sim_isaac_bridge] backend={args.backend} terrain={args.terrain}")
    print(f"[sim_isaac_bridge] depth.angle={getattr(env_cfg.depth, 'angle', None)}")
    print(f"[sim_isaac_bridge] action_scale={env_cfg.control.action_scale} "
          f"clip_actions={env_cfg.normalization.clip_actions} "
          f"-> policy_clip={env_cfg.normalization.clip_actions / env_cfg.control.action_scale}")
    print(f"[sim_isaac_bridge] steps={args.steps} num_envs={args.num_envs}")

    env, env_cfg = task_registry.make_env(name=ep_args.task, args=ep_args, env_cfg=env_cfg)
    # Print effective terrain after camera-headless rewrite
    props = list(env_cfg.terrain.terrain_proportions)
    keys = list(env_cfg.terrain.terrain_dict.keys())
    park = {k: env_cfg.terrain.terrain_dict[k] for k in keys if "parkour" in k or k == "demo"}
    print(f"[sim_isaac_bridge] actual num_envs={env.num_envs}")
    print(f"[sim_isaac_bridge] effective parkour-ish terrain={park}")
    print(f"[sim_isaac_bridge] depth.angle after make_env={env_cfg.depth.angle}")

    obs = env.get_observations()
    device = env.device

    depth_encoder = None
    policy_jit = None
    depth_actor = None

    if args.backend == "jit":
        print(f"[sim_isaac_bridge] base_jit={base_path}")
        print(f"[sim_isaac_bridge] vision_weight={vision_path}")
        depth_encoder = RecurrentDepthBackbone(n_proprio=N_PROPRIO).to(device)
        blob = torch.load(vision_path, map_location=device, weights_only=False)
        sd = blob["depth_encoder_state_dict"] if isinstance(blob, dict) and "depth_encoder_state_dict" in blob else blob
        depth_encoder.load_state_dict(sd, strict=True)
        depth_encoder.eval()
        depth_encoder.reset_hidden()
        policy_jit = torch.jit.load(base_path, map_location=device)
        policy_jit.eval()
    else:
        # evaluate.py path: load full runner, use depth_encoder + depth_actor
        log_pth = os.path.join("../../logs", ep_args.proj_name, ep_args.exptid)
        train_cfg.runner.resume = True
        ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(
            log_root=log_pth, env=env, name=ep_args.task, args=ep_args,
            train_cfg=train_cfg, return_log_dir=True, init_wandb=False,
        )
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=device)
        depth_actor = ppo_runner.alg.depth_actor
        depth_actor.eval()
        if hasattr(depth_encoder, "hidden_states"):
            depth_encoder.hidden_states = None
        print(f"[sim_isaac_bridge] ckpt log_pth={log_pth}")

    # Match evaluate.py: depth extras is only set every update_interval; else None.
    # Encoder/GRU must advance ONLY when a new depth frame arrives (~10 Hz), not every 50 Hz step.
    infos = {
        "depth": env.depth_buffer[:, -1].to(device) if env.cfg.depth.use_camera else None
    }
    latent = None
    yaw = None
    depth_updates = 0

    step_rew = []
    act_amax_raw = []
    vx_err = []
    rewbuffer = deque(maxlen=max(args.steps, 256))
    lenbuffer = deque(maxlen=max(args.steps, 256))
    wpbuffer = deque(maxlen=max(args.steps, 256))
    fallbuffer = deque(maxlen=max(args.steps, 256))

    cur_reward_sum = torch.zeros(env.num_envs, device=device)
    cur_episode_length = torch.zeros(env.num_envs, device=device)

    t0 = time.time()
    with torch.no_grad():
        for step in range(args.steps):
            if env.cfg.depth.use_camera:
                if infos.get("depth") is not None:
                    proprio = obs[:, :N_PROPRIO].clone()
                    proprio[:, 6:8] = 0
                    depth_out = depth_encoder(infos["depth"].to(device), proprio)
                    latent = depth_out[:, :-2]
                    yaw = depth_out[:, -2:]  # evaluate: scale applied when writing obs
                    depth_updates += 1
                # reuse previous latent/yaw when depth is None (same as evaluate/play)
                obs[:, 6:8] = YAW_SCALE * yaw
            else:
                latent = None

            if args.backend == "jit":
                actions = policy_jit(obs, latent)
            else:
                actions = depth_actor(obs.detach(), hist_encoding=True, scandots_latent=latent)

            act_amax_raw.append(float(actions.abs().max().item()))
            if args.preclamp:
                actions = actions.clamp(-ACTION_CLIP, ACTION_CLIP)

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
                if args.reset_gru_on_done:
                    ids = new_ids[:, 0]
                    if hasattr(depth_encoder, "reset_hidden_envs"):
                        depth_encoder.reset_hidden_envs(ids)
                    elif getattr(depth_encoder, "hidden_states", None) is not None:
                        depth_encoder.hidden_states[:, ids, :] = 0.0
            if killed_ids.numel() > 0:
                fallbuffer.extend([1.0] * killed_ids.shape[0])

            if step % args.log_every == 0 or step == args.steps - 1:
                print(
                    f"step={step:4d} rew_mean={step_rew[-1]:+.4f} "
                    f"amax_raw={act_amax_raw[-1]:.3f} "
                    f"vx_err={vx_err[-1]:.3f} "
                    f"vx_cmd={float(env.commands[0, 0]):.2f} "
                    f"vx={float(env.base_lin_vel[0, 0]):.2f} "
                    f"depth_upd={depth_updates} "
                    f"finite={bool(torch.isfinite(actions).all())}"
                )

    dt = time.time() - t0
    ep_count = len(rewbuffer)
    mxd = float(np.mean(np.array(wpbuffer, dtype=float) / WAYPOINT_NORM)) if wpbuffer else float("nan")
    mxd_std = float(np.std(np.array(wpbuffer, dtype=float) / WAYPOINT_NORM)) if len(wpbuffer) > 1 else 0.0
    len_mean = float(np.mean(lenbuffer)) if lenbuffer else float("nan")
    len_std = float(np.std(lenbuffer)) if len(lenbuffer) > 1 else 0.0
    rew_ep = float(np.mean(rewbuffer)) if rewbuffer else float("nan")
    fall_n = len(fallbuffer)
    print("--- summary ---")
    print(f"backend={args.backend} terrain={args.terrain}")
    print(f"steps={args.steps} envs={env.num_envs} wall_s={dt:.1f}")
    print(f"step_rew_mean={sum(step_rew)/len(step_rew):.4f}")
    print(f"vx_abs_err_mean={sum(vx_err)/len(vx_err):.4f}")
    print(f"amax_raw_max={max(act_amax_raw):.3f} amax_raw_mean={sum(act_amax_raw)/len(act_amax_raw):.3f}")
    print(f"episodes_finished={ep_count}")
    print(f"episode_rew_mean={rew_ep:.4f}")
    print(f"episode_len_mean={len_mean:.2f}±{len_std:.2f}")
    print(f"MXD(waypoints/7)={mxd:.4f}±{mxd_std:.4f}  (official Student ref ≈0.88)")
    print(f"falls_non_timeout={fall_n}")
    print(f"depth_encoder_forwards={depth_updates} (expect ~steps/update_interval)")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
