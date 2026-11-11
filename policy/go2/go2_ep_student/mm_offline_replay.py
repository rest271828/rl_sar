#!/usr/bin/env python3
"""Offline AB: recorded .mm depth -> EP preprocess -> encoder -> base_jit.

Implements valid%-gated depth handling for sim/offline (no robot required):
  fill   — always preprocess current frame (zeros -> 2 m), always update GRU
  hold   — if valid_frac < min_valid, blend holes from last-good frame, still update GRU
  refuse — if valid_frac < min_valid, skip encoder/GRU update (reuse prior latent/yaw)

Default min_valid=0.2 (matches ep_student::kDefaultMinValidFrac).

Example (110248 climb dump):
  conda run -n isaac_env --no-capture-output \\
    python policy/go2/go2_ep_student/mm_offline_replay.py \\
      --session logs/20261106_110248_go2_test --modes fill,hold,refuse
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ep_depth_encoder import RecurrentDepthBackbone  # noqa: E402

DEFAULT_MIN_VALID = 0.2
KEY_SEQS = {832, 851, 858, 865, 872, 918}


def preprocess_uint16(mm: np.ndarray, far: float = 2.0) -> np.ndarray:
    """Mirror C++ PreprocessDepthUint16Mm (nearest 106x60, crop, bilinear→58x87)."""
    h, w = mm.shape
    m = mm.astype(np.float32) / 1000.0
    m = np.where(mm > 0, m, far)
    m = np.clip(m, 0, far)
    ys = (np.linspace(0, h - 1, 60)).astype(int)
    xs = (np.linspace(0, w - 1, 106)).astype(int)
    small = m[ys][:, xs]
    crop = small[0:58, 4:102]
    t = torch.from_numpy(crop)[None, None]
    out = torch.nn.functional.interpolate(
        t, size=(58, 87), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    return out / far - 0.5


def valid_frac(mm: np.ndarray) -> float:
    return float((mm > 0).mean())


def apply_gate(
    mm: np.ndarray,
    prev_good: Optional[np.ndarray],
    mode: str,
    min_valid: float,
) -> Tuple[Optional[np.ndarray], str, float, bool]:
    """Returns (depth_58x87_or_None, tag, valid, encoder_updated).

    depth is None only when refuse skips update (caller keeps prior latent).
    """
    v = valid_frac(mm)
    if v >= min_valid or mode == "fill":
        return preprocess_uint16(mm), ("CUR" if v >= min_valid else "FILL_BAD"), v, True

    if mode == "hold":
        if prev_good is None:
            return preprocess_uint16(mm), "HOLD_NO_PREV", v, True
        use = prev_good.copy()
        m = mm > 0
        use[m] = mm[m]
        return preprocess_uint16(use), "HOLD", v, True

    if mode == "refuse":
        return None, "REFUSE", v, False

    raise ValueError(f"unknown mode {mode}")


def load_models(device: torch.device):
    enc = RecurrentDepthBackbone(n_proprio=53).to(device)
    blob = torch.load(_HERE / "vision_weight.pt", map_location=device, weights_only=False)
    sd = (
        blob["depth_encoder_state_dict"]
        if isinstance(blob, dict) and "depth_encoder_state_dict" in blob
        else blob
    )
    enc.load_state_dict(sd, strict=True)
    enc.eval()
    policy = torch.jit.load(str(_HERE / "base_jit.pt"), map_location=device)
    policy.eval()
    return enc, policy


def proprio_from_row(r: dict) -> torch.Tensor:
    p = torch.zeros(1, 53)
    p[0, 0] = float(r["gyro_x"])
    p[0, 1] = float(r["gyro_y"])
    p[0, 2] = float(r["gyro_z"])
    p[0, 10] = float(r["cmd_x"])
    p[0, 11] = 1.0  # parkour one-hot
    p[0, 12] = 0.0
    return p


def run_mode(
    mode: str,
    files: List[Path],
    ctrl: List[dict],
    rl0: int,
    rl1: int,
    enc: RecurrentDepthBackbone,
    policy,
    min_valid: float,
    print_all: bool,
) -> List[Dict]:
    def nearest(recv: int):
        return min(ctrl, key=lambda r: abs(int(r["t_ns"]) - recv))

    enc.reset_hidden()
    prev_good: Optional[np.ndarray] = None
    latent = torch.zeros(1, 32)
    yaw = torch.zeros(1, 2)
    obs = torch.zeros(1, 753)
    rows: List[Dict] = []

    print(f"==== mode={mode} min_valid={min_valid} ====")
    print("seq  RL_t  valid%  tag     upd  latent_l2  amax  front_max")

    for fp in files:
        meta = json.load(open(str(fp).replace(".mm", ".json")))
        d = np.fromfile(fp, dtype=np.uint16).reshape(meta["h"], meta["w"])
        in_rl = rl0 <= meta["recv_ns"] <= rl1
        if not in_rl:
            if valid_frac(d) >= min_valid:
                prev_good = d
            continue

        depth_np, tag, v, updated = apply_gate(d, prev_good, mode, min_valid)
        if v >= min_valid:
            prev_good = d

        row = nearest(meta["recv_ns"])
        proprio = proprio_from_row(row)
        with torch.no_grad():
            if updated:
                assert depth_np is not None
                out = enc(torch.from_numpy(depth_np)[None].float(), proprio)
                latent, yaw = out[:, :-2], out[:, -2:]
            # else: reuse latent/yaw; GRU hidden unchanged
            obs.zero_()
            obs[:, :53] = proprio
            obs[:, 6:8] = yaw * 1.5
            obs[:, 11] = 1.0
            obs[:, 12] = 0.0
            a = torch.clamp(policy(obs, latent), -4.8, 4.8)[0].numpy()

        rec = {
            "seq": meta["seq"],
            "t": (meta["recv_ns"] - rl0) / 1e9,
            "valid": v,
            "tag": tag,
            "updated": int(updated),
            "latent_l2": float(latent.norm().item()),
            "amax": float(np.abs(a).max()),
            "front_max": float(np.abs(a[[1, 2, 4, 5]]).max()),
        }
        rows.append(rec)
        if print_all or meta["seq"] in KEY_SEQS or tag in ("HOLD", "REFUSE", "FILL_BAD"):
            print(
                f"seq{rec['seq']:4d} {rec['t']:5.2f}  {rec['valid']*100:5.1f}  "
                f"{rec['tag']:8s}  {rec['updated']}  {rec['latent_l2']:6.3f}  "
                f"{rec['amax']:5.3f}  {rec['front_max']:5.3f}"
            )
    return rows


def summarize(mode: str, rows: List[Dict]) -> None:
    by_seq = {r["seq"]: r for r in rows}
    print(f"-- summary {mode} (key frames) --")
    for s in sorted(KEY_SEQS):
        if s not in by_seq:
            continue
        r = by_seq[s]
        print(
            f"  seq{s}: valid={r['valid']*100:.1f}% tag={r['tag']} "
            f"upd={r['updated']} latent_l2={r['latent_l2']:.3f} amax={r['amax']:.3f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--session",
        default="logs/20261106_110248_go2_test",
        help="session dir relative to repo root or absolute (must contain control.csv + depth/)",
    )
    ap.add_argument(
        "--modes",
        default="fill,hold,refuse",
        help="comma list: fill,hold,refuse",
    )
    ap.add_argument("--min-valid", type=float, default=DEFAULT_MIN_VALID)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--all-frames", action="store_true")
    args = ap.parse_args()

    repo = _HERE.parents[2]  # .../rl_sar_ep
    sess = Path(args.session)
    if not sess.is_absolute():
        sess = repo / sess
    if not (sess / "control.csv").is_file():
        raise SystemExit(f"missing {sess}/control.csv")

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    enc, policy = load_models(device)
    print(f"models_ok device={device} session={sess}")

    ctrl = [
        r
        for r in csv.DictReader(open(sess / "control.csv"))
        if r["fsm"] == "RLFSMStateRLLocomotion"
    ]
    if not ctrl:
        raise SystemExit("no RLFSMStateRLLocomotion rows")
    rl0, rl1 = int(ctrl[0]["t_ns"]), int(ctrl[-1]["t_ns"])
    files = sorted((sess / "depth").glob("*.mm"))
    if not files:
        raise SystemExit(f"no .mm under {sess}/depth")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_rows = {}
    for mode in modes:
        rows = run_mode(
            mode, files, ctrl, rl0, rl1, enc, policy, args.min_valid, args.all_frames
        )
        summarize(mode, rows)
        all_rows[mode] = rows

    # Before/after on seq858
    if 858 in {r["seq"] for rows in all_rows.values() for r in rows}:
        print("==== seq858 before/after (bad frame) ====")
        for mode, rows in all_rows.items():
            r = next(x for x in rows if x["seq"] == 858)
            print(
                f"  {mode:6s}: tag={r['tag']:8s} encoder_updated={r['updated']} "
                f"latent_l2={r['latent_l2']:.3f} amax={r['amax']:.3f}"
            )
    print("DONE")
    return 0


if __name__ == "__main__":
    # Allow override via env for one-liners
    if "RL_SAR_EP_DEPTH_MIN_VALID" in os.environ and "--min-valid" not in sys.argv:
        sys.argv.extend(["--min-valid", os.environ["RL_SAR_EP_DEPTH_MIN_VALID"]])
    raise SystemExit(main())
