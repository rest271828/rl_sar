#!/usr/bin/env python3
"""Offline dry-run for Extreme Parkour Go2 Student on the rl_sar policy slot.

Loads base_jit + vision_weight, runs dummy depth+obs, prints 12-dim actions.
Depth GRU advances only every update_interval (=5) steps; other steps reuse latent (deploy cadence).
No robot / ROS required.

Example:
  source /home/yihan/extreme-parkour/activate.sh
  python policy/go2/go2_ep_student/infer_offline.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Allow running as script from repo root or from this directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ep_depth_encoder import RecurrentDepthBackbone

DEFAULT_TRACED = (
    "/home/yihan/extreme-parkour/extreme-parkour-repro/legged_gym/logs/"
    "parkour_new/922-10-GO2-STUDENT-D0/traced"
)
DEFAULT_BASE = os.path.join(DEFAULT_TRACED, "922-10-GO2-STUDENT-D0-59500-base_jit.pt")
DEFAULT_VISION = os.path.join(DEFAULT_TRACED, "922-10-GO2-STUDENT-D0-59500-vision_weight.pt")

N_PROPRIO = 53
N_OBS = 753
DEPTH_H, DEPTH_W = 58, 87
ACTION_CLIP = 4.8
YAW_SCALE = 1.5


def resolve_weight(local_name: str, default_abs: str) -> str:
    local = os.path.join(_HERE, local_name)
    if os.path.isfile(local):
        return local
    if os.path.isfile(default_abs):
        return default_abs
    raise FileNotFoundError(
        f"Missing weight: tried {local} and {default_abs}. "
        "Copy traced *.pt into this directory or pass --base-jit / --vision-weight."
    )


def build_encoder(vision_path: str, device: torch.device) -> RecurrentDepthBackbone:
    enc = RecurrentDepthBackbone(n_proprio=N_PROPRIO).to(device)
    blob = torch.load(vision_path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "depth_encoder_state_dict" in blob:
        sd = blob["depth_encoder_state_dict"]
    else:
        sd = blob
    missing, unexpected = enc.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"vision load mismatch missing={missing} unexpected={unexpected}")
    enc.eval()
    return enc


def main() -> int:
    ap = argparse.ArgumentParser(description="EP Go2 Student offline inference dry-run")
    ap.add_argument("--base-jit", default=None, help="path to *-base_jit.pt")
    ap.add_argument("--vision-weight", default=None, help="path to *-vision_weight.pt")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=3, help="GRU steps with dummy depth")
    ap.add_argument("--obs-mode", choices=["zeros", "ones"], default="zeros")
    ap.add_argument("--depth-mode", choices=["zeros", "ones", "mid"], default="mid")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    base_path = args.base_jit or resolve_weight("base_jit.pt", DEFAULT_BASE)
    vision_path = args.vision_weight or resolve_weight("vision_weight.pt", DEFAULT_VISION)

    print(f"device={device}")
    print(f"base_jit={base_path}")
    print(f"vision_weight={vision_path}")

    encoder = build_encoder(vision_path, device)
    policy = torch.jit.load(base_path, map_location=device)
    policy.eval()

    if args.obs_mode == "zeros":
        obs = torch.zeros(1, N_OBS, device=device)
    else:
        obs = torch.ones(1, N_OBS, device=device)

    # Flat-ground-ish proprio hints: terrain one-hot parkour [1,0] at indices 11–12
    obs[:, 11] = 1.0
    obs[:, 12] = 0.0

    if args.depth_mode == "zeros":
        depth = torch.zeros(1, DEPTH_H, DEPTH_W, device=device)
    elif args.depth_mode == "ones":
        depth = torch.ones(1, DEPTH_H, DEPTH_W, device=device)
    else:
        # Normalized depth mid-range ~0 (EP preprocess centers around 0)
        depth = torch.zeros(1, DEPTH_H, DEPTH_W, device=device)

    encoder.reset_hidden()
    last_action = None
    update_interval = 5  # match env.cfg.depth.update_interval / evaluate.py
    latent = torch.zeros(1, 32, device=device)
    yaw = torch.zeros(1, 2, device=device)
    with torch.no_grad():
        for t in range(args.steps):
            # Advance depth GRU only on the deploy cadence (not every control step).
            if t % update_interval == 0:
                proprio = obs[:, :N_PROPRIO].clone()
                proprio[:, 6:8] = 0
                depth_out = encoder(depth, proprio)  # [1, 34]
                latent = depth_out[:, :-2]
                yaw = depth_out[:, -2:]
            obs[:, 6:8] = YAW_SCALE * yaw
            action = policy(obs.clone(), latent)
            action_clipped = action.clamp(-ACTION_CLIP, ACTION_CLIP)
            last_action = action_clipped
            a = action_clipped.flatten()
            print(
                f"step={t} action_dim={tuple(action.shape)} "
                f"amax={float(a.abs().max()):.4f} "
                f"mean={float(a.mean()):.4f} "
                f"yaw={yaw.flatten().tolist()} "
                f"latent_norm={float(latent.norm()):.4f} "
                f"finite={bool(torch.isfinite(action).all())}"
            )
            print(f"  action[12]={[round(float(x), 4) for x in a.tolist()]}")

    assert last_action is not None and last_action.shape[-1] == 12
    assert bool(torch.isfinite(last_action).all())
    print("OK: weights loaded; 12-dim finite actions produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
