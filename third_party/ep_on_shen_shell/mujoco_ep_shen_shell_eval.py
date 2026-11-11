#!/usr/bin/env python3
"""EP Student on Shen rl_sar / MuJoCo shell — plane depth, stable_isaac_v6 preferred.

Uses rl_sar go2_ep_student weights + scenes/go2_ep_step40 stable_v6 XML.
Plane-depth closed loop shared with mujoco_contact_tune (prior EP Python loop).
Not himloco / dreamwaq.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

import os

HERE = Path(__file__).resolve().parent
# Laptop layout: third_party/{ep_on_shen_shell,mujoco_contact_tune,damping_isaac_v6}
# Override with RL_SAR_ROOT if needed.
CONTACT = HERE.parent / "mujoco_contact_tune"
FRONT = HERE.parent / "damping_isaac_v6"
_DEFAULT_RL = HERE.parents[1]  # .../rl_sar_ep
if not (_DEFAULT_RL / "policy/go2/go2_ep_student").is_dir():
    _DEFAULT_RL = Path("/home/yihan/extreme-parkour/rest271828-rl_sar")
RL_SAR = Path(os.environ.get("RL_SAR_ROOT", str(_DEFAULT_RL))).expanduser().resolve()
EP_POLICY = RL_SAR / "policy/go2/go2_ep_student"
SHELL_SCENE = RL_SAR / "scenes/go2_ep_step40/scene_step40_stable_v6.xml"

sys.path.insert(0, str(CONTACT))
import mujoco_step40_eval as mj  # noqa: E402

sys.path.insert(0, str(FRONT))
import mujoco_step40_eval_front as mj_front  # noqa: E402


def load_ep_shell_meta() -> dict:
    raw = yaml.safe_load((EP_POLICY / "config.yaml").read_text())
    assert isinstance(raw, dict) and len(raw) == 1
    key, cfg = next(iter(raw.items()))
    vision_name = cfg.get("vision_weight_name", "vision_weight.pt")
    vision_path = EP_POLICY / vision_name
    # C++ uses EPWT .bin; Python encoder needs torch .pt
    if vision_path.suffix == ".bin" or not vision_path.is_file():
        alt = EP_POLICY / "vision_weight.pt"
        if alt.is_file():
            vision_path = alt
    return {
        "config_key": key,
        "model": str(EP_POLICY / cfg["model_name"]),
        "vision": str(vision_path),
        "ep_num_observations": cfg.get("ep_num_observations"),
        "ep_depth_latent_dim": cfg.get("ep_depth_latent_dim"),
        "ep_use_dual_input": cfg.get("ep_use_dual_input"),
        "ep_update_interval": cfg.get("ep_update_interval"),
        "rl_kp": cfg.get("rl_kp"),
        "rl_kd": cfg.get("rl_kd"),
        "action_scale": cfg.get("action_scale"),
        "joint_mapping": cfg.get("joint_mapping"),
        "note": (
            "Closed-loop uses EP Unitree FR/FL/RR/RL order (plane-depth contract). "
            "yaml joint_mapping is himloco-shaped for InitRL sizing; not re-applied here."
        ),
    }


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r["config"], {}).setdefault(r["cmd"], []).append(r)
    out = {}
    for cfg, cmds in by.items():
        out[cfg] = {}
        for cmd, trials in cmds.items():
            n = len(trials)
            out[cfg][cmd] = {
                "n": n,
                "pass": sum(t["passed"] for t in trials),
                "pass_rate": sum(t["passed"] for t in trials) / n,
                "max_x_mean": float(np.mean([t["max_x"] for t in trials])),
                "best_streak_mean": float(np.mean([t["best_streak"] for t in trials])),
                "joint_vel_rms_mean": float(np.nanmean([t["joint_vel_rms"] for t in trials])),
                "calf_vel_rms_mean": float(np.nanmean([t["calf_vel_rms"] for t in trials])),
                "angvel_rms_mean": float(np.nanmean([t["angvel_rms"] for t in trials])),
                "swing_clear_p95_mean": float(np.nanmean([t["swing_clear_p95"] for t in trials])),
                "post_climb_hold_mean": float(np.mean([t["post_climb_hold"] for t in trials])),
                "first_latent_norm_mean": float(np.nanmean([t["first_latent_norm"] for t in trials])),
                "depth_updates_mean": float(np.mean([t["depth_updates"] for t in trials])),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=SHELL_SCENE)
    ap.add_argument("--config", type=str, default="stable_isaac_v6")
    ap.add_argument("--cmd", type=float, default=1.0)
    ap.add_argument("--cmds", type=str, default="0.8,1.0,1.1,1.2,1.3")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--dump-actions", type=Path, default=None, help="CSV of per-step actions")
    args = ap.parse_args()

    if args.config in mj.CONFIGS:
        contact_mod = mj
    elif args.config in mj_front.CONFIGS:
        # stable_isaac_v6 lives in the front-leg module; same scene, no new grid.
        contact_mod = mj_front
    else:
        raise SystemExit(f"unknown contact config {args.config}")
    if not args.scene.exists():
        raise SystemExit(f"missing shell scene {args.scene}")

    meta = load_ep_shell_meta()
    jit = Path(meta["model"])
    vision = Path(meta["vision"])
    if not jit.exists() or not vision.exists():
        raise SystemExit(f"missing weights jit={jit.exists()} vision={vision.exists()}")

    contact = contact_mod.CONFIGS[args.config]
    cmds = [float(x) for x in args.cmds.split(",")] if args.grid else [args.cmd]
    args.out.mkdir(parents=True, exist_ok=True)
    vid_dir = HERE / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    print("== EP on Shen rl_sar shell ==")
    print(f"scene={args.scene}")
    print(f"contact={args.config} note={contact.note}")
    print(f"policy={EP_POLICY} jit={jit.name} vision={vision.name}")
    print("depth=plane update_interval=%s" % (meta.get("ep_update_interval"),))

    all_rows = []
    t0 = time.time()
    for cmd in cmds:
        for trial in range(args.trials):
            rec = None
            if args.record and trial == 0:
                cmd_tag = f"{cmd:.1f}".replace(".", "")
                suffix = f"-{args.tag}" if args.tag else ""
                # deliverable names without extra tag for main vids
                if args.tag:
                    rec = vid_dir / f"ep-shen-shell-gait{suffix}-cmd{cmd_tag}.mp4"
                else:
                    rec = vid_dir / f"ep-shen-shell-gait-cmd{cmd_tag}.mp4"
            model = contact_mod.load_model(args.scene, contact)
            policy = mj.EPStudentPolicy(vision, jit, device=args.device)
            dump_path = None
            if args.dump_actions is not None and trial == 0 and cmd == cmds[0]:
                dump_path = args.dump_actions
            res = mj.run_episode(
                model,
                policy,
                cmd,
                f"shell/{args.config}",
                trial,
                n_steps=args.steps,
                record_path=rec,
                cfg=contact,
                depth_backend="plane",
                dump_actions_path=dump_path,
            )
            d = asdict(res)
            d["shell"] = "rl_sar/go2_ep_student"
            d["scene"] = str(args.scene)
            d["depth"] = "plane"
            if rec is not None:
                d["video"] = str(rec)
            all_rows.append(d)
            print(
                f"[EP-shell] cmd={cmd:.1f} t={trial} pass={res.passed} "
                f"streak={res.best_streak} max_x={res.max_x:.2f} "
                f"jvel={res.joint_vel_rms:.2f} calf={res.calf_vel_rms:.2f} "
                f"hold={res.post_climb_hold} latent0={res.first_latent_norm:.3f} "
                f"depth_u={res.depth_updates}"
                + (f" vid={rec.name}" if rec else ""),
                flush=True,
            )

    summary = summarize(all_rows)
    stamp = args.tag or "run"
    out_json = args.out / f"ep_shell_{stamp}_{int(time.time())}.json"
    payload = {
        "meta": {
            "shell": "rest271828-rl_sar / go2_ep_student",
            "scene": str(args.scene),
            "contact": args.config,
            "depth": "plane",
            "cmds": cmds,
            "trials": args.trials,
            "steps": args.steps,
            "elapsed_s": time.time() - t0,
            "pass_streak": mj.PASS_STREAK,
            "ep_shell_meta": meta,
            "prior_python_loop": str(CONTACT / "mujoco_step40_eval.py"),
            "not": "himloco/dreamwaq",
        },
        "rows": all_rows,
        "summary": summary,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print("wrote", out_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
