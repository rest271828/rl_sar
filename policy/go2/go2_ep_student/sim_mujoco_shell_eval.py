#!/usr/bin/env python3
"""Thin launcher: EP Student MuJoCo shell eval (plane depth / stable_isaac_v6).

Resolution order:
1) A6000 original path (if present on this machine)
2) Dropped tree from user pack: third_party/ep_on_shen_shell/mujoco_ep_shen_shell_eval.py
3) In-repo local shell mujoco_ep_shell_eval.py
"""
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_CANDIDATES = [
    Path("/home/yihan/agent_ws/experiments/ep_on_shen_shell/mujoco_ep_shen_shell_eval.py"),
    _REPO / "third_party" / "ep_on_shen_shell" / "mujoco_ep_shen_shell_eval.py",
    _HERE / "mujoco_ep_shell_eval.py",
]
TARGET = next((p for p in _CANDIDATES if p.is_file()), _CANDIDATES[-1])
print(f"[sim_mujoco_shell_eval] using {TARGET}", file=sys.stderr)
runpy.run_path(str(TARGET), run_name="__main__")
