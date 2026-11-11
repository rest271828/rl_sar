#!/usr/bin/env python3
"""Driver: diagnose + A/B front-leg climb fixes on EP/Shen shell scene."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mujoco_step40_eval_front as mj  # noqa: E402

import os as _os
_HERE = Path(__file__).resolve().parent
_DEFAULT_RL = _HERE.parents[1]  # third_party -> rl_sar_ep
if not (_DEFAULT_RL / "policy/go2/go2_ep_student").is_dir():
    _DEFAULT_RL = Path("/home/yihan/extreme-parkour/rest271828-rl_sar")
RL_SAR = Path(_os.environ.get("RL_SAR_ROOT", str(_DEFAULT_RL))).expanduser().resolve()
SHELL_SCENE = RL_SAR / "scenes/go2_ep_step40/scene_step40_stable_v6.xml"
JIT = RL_SAR / "policy/go2/go2_ep_student/base_jit.pt"
VISION = RL_SAR / "policy/go2/go2_ep_student/vision_weight.pt"


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r["config"], {}).setdefault(r["cmd"], []).append(r)
    out = {}
    for cfg, cmds in by.items():
        out[cfg] = {}
        for cmd, trials in cmds.items():
            n = len(trials)
            def m(key):
                return float(np.nanmean([t[key] for t in trials]))

            out[cfg][cmd] = {
                "n": n,
                "pass": sum(int(t["passed"]) for t in trials),
                "pass_rate": sum(int(t["passed"]) for t in trials) / n,
                "best_streak_mean": m("best_streak"),
                "post_climb_hold_mean": m("post_climb_hold"),
                "joint_vel_rms_mean": m("joint_vel_rms"),
                "calf_vel_rms_mean": m("calf_vel_rms"),
                "front_calf_rms_mean": m("front_calf_rms"),
                "rear_calf_rms_mean": m("rear_calf_rms"),
                "front_swing_p95_mean": m("front_swing_p95"),
                "rear_swing_p95_mean": m("rear_swing_p95"),
                "approach_front_calf_rms": m("approach_front_calf_rms"),
                "approach_rear_calf_rms": m("approach_rear_calf_rms"),
                "approach_front_swing_p95": m("approach_front_swing_p95"),
                "approach_front_jvel_rms": m("approach_front_jvel_rms"),
                "approach_rear_jvel_rms": m("approach_rear_jvel_rms"),
                "approach_pitch_rms": m("approach_pitch_rms"),
                "preclimb_front_calf_rms": m("preclimb_front_calf_rms"),
                "preclimb_rear_calf_rms": m("preclimb_rear_calf_rms"),
                "preclimb_front_swing_p95": m("preclimb_front_swing_p95"),
                "preclimb_rear_swing_p95": m("preclimb_rear_swing_p95"),
                "preclimb_front_jvel_rms": m("preclimb_front_jvel_rms"),
                "preclimb_rear_jvel_rms": m("preclimb_rear_jvel_rms"),
                "preclimb_pitch_rms": m("preclimb_pitch_rms"),
                "ondeck_front_calf_rms": m("ondeck_front_calf_rms"),
                "ondeck_rear_calf_rms": m("ondeck_rear_calf_rms"),
                "ondeck_front_swing_p95": m("ondeck_front_swing_p95"),
                "ondeck_pitch_rms": m("ondeck_pitch_rms"),
                "swing_clear_p95_mean": m("swing_clear_p95"),
                "swing_clear_mean_mean": m("swing_clear_mean"),
                "joint_vel_peak_mean": m("joint_vel_peak"),
                "action_rate_rms_mean": m("action_rate_rms"),
            }
            # ratios
            af, ar = out[cfg][cmd]["approach_front_calf_rms"], out[cfg][cmd]["approach_rear_calf_rms"]
            pf, pr = out[cfg][cmd]["preclimb_front_calf_rms"], out[cfg][cmd]["preclimb_rear_calf_rms"]
            out[cfg][cmd]["approach_front_rear_calf_ratio"] = (
                af / ar if ar and ar == ar and ar > 1e-9 else float("nan")
            )
            out[cfg][cmd]["preclimb_front_rear_calf_ratio"] = (
                pf / pr if pr and pr == pr and pr > 1e-9 else float("nan")
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, default="stable_isaac_v6")
    ap.add_argument("--cmds", type=str, default="0.8,1.0,1.1,1.2,1.3")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--scene", type=Path, default=SHELL_SCENE)
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    cmds = [float(x) for x in args.cmds.split(",")]
    for c in configs:
        if c not in mj.CONFIGS:
            raise SystemExit(f"unknown config {c}")

    args.out.mkdir(parents=True, exist_ok=True)
    vid_dir = HERE / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for cname in configs:
        cfg = mj.CONFIGS[cname]
        print(f"== {cname}: {cfg.note}")
        for cmd in cmds:
            for trial in range(args.trials):
                rec = None
                if args.record and trial == 0:
                    tag = f"{cmd:.1f}".replace(".", "")
                    rec = vid_dir / f"frontleg-{cname}-cmd{tag}.mp4"
                model = mj.load_model(args.scene, cfg)
                policy = mj.EPStudentPolicy(VISION, JIT, device=args.device)
                res = mj.run_episode(
                    model,
                    policy,
                    cmd,
                    cname,
                    trial,
                    n_steps=args.steps,
                    record_path=rec,
                    cfg=cfg,
                )
                d = asdict(res)
                if rec is not None:
                    d["video"] = str(rec)
                rows.append(d)
                print(
                    f"[{cname}] cmd={cmd:.1f} t={trial} pass={res.passed} "
                    f"streak={res.best_streak} jvel={res.joint_vel_rms:.2f} calf={res.calf_vel_rms:.2f} "
                    f"apF/R={res.approach_front_calf_rms:.2f}/{res.approach_rear_calf_rms:.2f} "
                    f"apSwF95={res.approach_front_swing_p95:.3f} "
                    f"preF/R={res.preclimb_front_calf_rms:.2f}/{res.preclimb_rear_calf_rms:.2f} "
                    f"preSwF95={res.preclimb_front_swing_p95:.3f} "
                    f"onF={res.ondeck_front_calf_rms:.2f} onPitch={res.ondeck_pitch_rms:.3f}"
                )

    summary = summarize(rows)
    out_json = args.out / f"results_{args.tag}.json"
    sum_json = args.out / f"summary_{args.tag}.json"
    out_json.write_text(json.dumps({"results": rows, "elapsed_s": time.time() - t0}, indent=2))
    sum_json.write_text(json.dumps(summary, indent=2))
    print("wrote", out_json)
    print("wrote", sum_json)
    # compact table
    print("\n== compact ==")
    for cname, cmds_d in summary.items():
        passes = sum(v["pass"] for v in cmds_d.values())
        total = sum(v["n"] for v in cmds_d.values())
        # mean over cmds of key metrics
        keys = [
            "approach_front_calf_rms",
            "approach_front_swing_p95",
            "preclimb_front_swing_p95",
            "preclimb_front_calf_rms",
            "ondeck_front_calf_rms",
            "joint_vel_rms_mean",
            "calf_vel_rms_mean",
            "swing_clear_p95_mean",
            "joint_vel_peak_mean",
        ]
        means = {k: float(np.nanmean([v[k] for v in cmds_d.values()])) for k in keys}
        print(
            f"{cname}: pass={passes}/{total} "
            f"apFcalf={means['approach_front_calf_rms']:.2f} "
            f"apSwF={means['approach_front_swing_p95']:.3f} "
            f"preSwF={means['preclimb_front_swing_p95']:.3f} "
            f"onF={means['ondeck_front_calf_rms']:.2f} "
            f"jvel={means['joint_vel_rms_mean']:.2f} calf={means['calf_vel_rms_mean']:.2f} "
            f"swp95={means['swing_clear_p95_mean']:.3f} jpeak={means['joint_vel_peak_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
