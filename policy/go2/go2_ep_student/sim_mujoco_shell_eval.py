#!/usr/bin/env python3
"""Thin launcher: EP Student MuJoCo shell eval (plane depth / stable_isaac_v6).

Historically this redirected to the A6000-only path
`/home/yihan/agent_ws/experiments/ep_on_shen_shell/mujoco_ep_shen_shell_eval.py`.
That file is not on this laptop (and SSH to A6000 currently fails banner exchange),
so we run the in-repo local shell instead.
"""
import runpy
from pathlib import Path

_LOCAL = Path(__file__).resolve().parent / "mujoco_ep_shell_eval.py"
_A6000 = Path("/home/yihan/agent_ws/experiments/ep_on_shen_shell/mujoco_ep_shen_shell_eval.py")
TARGET = _A6000 if _A6000.is_file() else _LOCAL
runpy.run_path(str(TARGET), run_name="__main__")
