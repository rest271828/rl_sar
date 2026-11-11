#!/usr/bin/env python3
"""MuJoCo Go2 EP Student closed-loop on scene_step40 (plane depth).

Restores the plane-depth contract used by the Oct 12–14 rl_sar splice, and
exposes contact/dynamics levers for A/B toward Isaac training distribution.

Usage (on A6000, extreme-parkour env):
  source /home/yihan/extreme-parkour/activate.sh
  python mujoco_step40_eval.py --cmd 1.2 --config baseline --trials 1
  python mujoco_step40_eval.py --grid --config friction1.0 --trials 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mujoco
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths / weights
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
import os as _os
_HERE = Path(__file__).resolve().parent
_DEFAULT_RL = _HERE.parents[1]  # third_party -> rl_sar_ep
if not (_DEFAULT_RL / "policy/go2/go2_ep_student").is_dir():
    _DEFAULT_RL = Path("/home/yihan/extreme-parkour/rest271828-rl_sar")
RL_SAR = Path(_os.environ.get("RL_SAR_ROOT", str(_DEFAULT_RL))).expanduser().resolve()
EP_POLICY = RL_SAR / "policy/go2/go2_ep_student"
DEFAULT_SCENE = HERE / "scene_step40.xml"
DEFAULT_VISION = EP_POLICY / "vision_weight.pt"
DEFAULT_JIT = EP_POLICY / "base_jit.pt"
sys.path.insert(0, str(EP_POLICY))

from ep_depth_encoder import RecurrentDepthBackbone  # noqa: E402

# Policy / MuJoCo actuator order: FR, FL, RR, RL (hip, thigh, calf)
DEFAULT_DOF = np.array(
    [-0.1, 0.7, -1.5, 0.1, 0.7, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5],
    dtype=np.float64,
)
FOOT_GEOM_NAMES = ["FR", "FL", "RR", "RL"]  # contact order in obs[49:53]
# qpos/qvel indices for policy order FR,FL,RR,RL (XML joint tree is FL,FR,RL,RR)
QPOS_IDX = [10, 11, 12, 7, 8, 9, 16, 17, 18, 13, 14, 15]
QVEL_IDX = [9, 10, 11, 6, 7, 8, 15, 16, 17, 12, 13, 14]

N_PROPRIO = 53
N_SCAN = 132
N_PRIV_E = 9
N_PRIV_L = 29
HIST_LEN = 10
N_OBS = N_PROPRIO + N_SCAN + N_PRIV_E + N_PRIV_L + HIST_LEN * N_PROPRIO  # 753

DT = 0.005
DECIMATION = 4
UPDATE_INTERVAL = 5
ACTION_DELAY = 1
ACTION_SCALE = 0.25
ACTION_CLIP = 4.8
KP = 40.0
KD = 1.0
TORQUE_LIMIT = np.array(
    [23.7, 23.7, 45.43] * 4, dtype=np.float64
)  # hip, thigh, calf per leg in FR/FL/RR/RL

# Depth camera (EP Go2). Training draws pitch once per env in attach_camera,
# uniform on this range. play.py does not redraw it every step (A1 play forces
# [0, 1]; Go2 play keeps this range). Eval holds the midpoint for the episode.
CAM_POS_BODY = np.array([0.24, -0.0175, 0.12], dtype=np.float64)
CAM_PITCH_RANGE_DEG = (21.2, 24.6)
CAM_PITCH_DEG = 0.5 * (CAM_PITCH_RANGE_DEG[0] + CAM_PITCH_RANGE_DEG[1])
HFOV_DEG = 87.0
DEPTH_ORIGIN = (106, 60)  # W, H
DEPTH_RESIZED = (87, 58)  # W, H
NEAR_CLIP = 0.0
FAR_CLIP = 2.0

# Pass criterion (docs)
PASS_X = (1.20, 3.30)
PASS_Z = (0.64, 0.95)
PASS_ROLL = 0.5
PASS_PITCH = 0.6
PASS_STREAK = 50


# ---------------------------------------------------------------------------
# Contact configs (one lever at a time from stock go2.xml)
# ---------------------------------------------------------------------------

@dataclass
class ContactConfig:
    name: str
    note: str
    foot_friction: Optional[Tuple[float, float, float]] = None  # None = leave XML
    floor_friction: Optional[Tuple[float, float, float]] = None
    step_friction: Optional[Tuple[float, float, float]] = None
    armature: Optional[float] = None
    margin: Optional[float] = None
    solref: Optional[Tuple[float, float]] = None
    solimp: Optional[Tuple[float, float, float, float, float]] = None
    impratio: Optional[float] = None
    timestep: Optional[float] = None
    joint_frictionloss: Optional[float] = None
    joint_damping: Optional[float] = None
    # Control / timing (defaults match Isaac Go2ParkourCfg)
    kp: Optional[float] = None
    kd: Optional[float] = None
    decimation: Optional[int] = None
    action_scale: Optional[float] = None
    # v3 flail levers
    foot_condim: Optional[int] = None
    cone: Optional[str] = None  # "elliptic" | "pyramidal"
    foot_size: Optional[float] = None  # sphere radius (stock 0.022)
    action_ema: Optional[float] = None  # blend new action: ema*new + (1-ema)*prev
    torque_scale: Optional[float] = None  # multiply TORQUE_LIMIT
    # v4: calf-only viscous damping (applied AFTER joint_damping if both set)
    calf_damping: Optional[float] = None
    # v5 front-leg climb levers (sim-only; plane depth unchanged)
    front_calf_damping: Optional[float] = None
    front_thigh_damping: Optional[float] = None
    front_hip_damping: Optional[float] = None
    front_kd_scale: Optional[float] = None
    front_action_scale: Optional[float] = None
    # GO2-32: further damping / contact levers on fc38 lineage
    thigh_damping: Optional[float] = None  # all thigh hinges
    hip_damping: Optional[float] = None    # all hip hinges
    rear_calf_damping: Optional[float] = None
    rear_thigh_damping: Optional[float] = None
    rear_hip_damping: Optional[float] = None


CONFIGS: Dict[str, ContactConfig] = {
    "baseline": ContactConfig(
        name="baseline",
        note="Stock go2.xml: foot fri=0.4/0.02/0.01, armature=0.01, margin=0.001, "
        "cone=elliptic impratio=100, MuJoCo default solref/solimp",
    ),
    "friction1.0": ContactConfig(
        name="friction1.0",
        note="Foot+floor+step sliding friction → 1.0 (Isaac static/dynamic=1.0)",
        foot_friction=(1.0, 0.005, 0.0001),
        floor_friction=(1.0, 0.005, 0.0001),
        step_friction=(1.0, 0.005, 0.0001),
    ),
    "friction0.8": ContactConfig(
        name="friction0.8",
        note="Friction 0.8 (low end of Isaac domain-rand [0.6,2])",
        foot_friction=(0.8, 0.005, 0.0001),
        floor_friction=(0.8, 0.005, 0.0001),
        step_friction=(0.8, 0.005, 0.0001),
    ),
    "friction1.2": ContactConfig(
        name="friction1.2",
        note="Friction 1.2 (mid Isaac domain-rand)",
        foot_friction=(1.2, 0.005, 0.0001),
        floor_friction=(1.2, 0.005, 0.0001),
        step_friction=(1.2, 0.005, 0.0001),
    ),
    "armature0": ContactConfig(
        name="armature0",
        note="Joint armature 0.01 → 0.0 (Isaac asset.armature=0)",
        armature=0.0,
    ),
    "margin0.01": ContactConfig(
        name="margin0.01",
        note="Geom margin 0.001 → 0.01 (Isaac physx.contact_offset=0.01)",
        margin=0.01,
    ),
    "solref_soft": ContactConfig(
        name="solref_soft",
        note="Softer contacts solref=(0.04,1) vs default (0.02,1)",
        solref=(0.04, 1.0),
    ),
    "solref_stiff": ContactConfig(
        name="solref_stiff",
        note="Stiffer contacts solref=(0.01,1)",
        solref=(0.01, 1.0),
    ),
    "solimp_soft": ContactConfig(
        name="solimp_soft",
        note="Softer solimp dwidth/width (0.9 0.95 0.001 → 0.8 0.9 0.01)",
        solimp=(0.8, 0.9, 0.01, 0.5, 2.0),
    ),
    "impratio1": ContactConfig(
        name="impratio1",
        note="impratio 100 → 1 (less elliptic torsional friction emphasis)",
        impratio=1.0,
    ),
    # Combined: friction toward Isaac + armature 0 (best-of singles if both help)
    "fri1_arm0": ContactConfig(
        name="fri1_arm0",
        note="Combine friction1.0 + armature0",
        foot_friction=(1.0, 0.005, 0.0001),
        floor_friction=(1.0, 0.005, 0.0001),
        step_friction=(1.0, 0.005, 0.0001),
        armature=0.0,
    ),
    "fri1_margin01": ContactConfig(
        name="fri1_margin01",
        note="friction1.0 + margin 0.01",
        foot_friction=(1.0, 0.005, 0.0001),
        floor_friction=(1.0, 0.005, 0.0001),
        step_friction=(1.0, 0.005, 0.0001),
        margin=0.01,
    ),
    "fri1_arm0_margin01": ContactConfig(
        name="fri1_arm0_margin01",
        note="friction1.0 + armature0 + margin0.01",
        foot_friction=(1.0, 0.005, 0.0001),
        floor_friction=(1.0, 0.005, 0.0001),
        step_friction=(1.0, 0.005, 0.0001),
        armature=0.0,
        margin=0.01,
    ),
    "fri12_margin01": ContactConfig(
        name="fri12_margin01",
        note="Best single friction1.2 + margin0.01 (margin alone rescued cmd1.2)",
        foot_friction=(1.2, 0.005, 0.0001),
        floor_friction=(1.2, 0.005, 0.0001),
        step_friction=(1.2, 0.005, 0.0001),
        margin=0.01,
    ),
    "fri12_solref_stiff": ContactConfig(
        name="fri12_solref_stiff",
        note="friction1.2 + stiffer solref",
        foot_friction=(1.2, 0.005, 0.0001),
        floor_friction=(1.2, 0.005, 0.0001),
        step_friction=(1.2, 0.005, 0.0001),
        solref=(0.01, 1.0),
    ),
    "fri15": ContactConfig(
        name="fri15",
        note="Friction 1.5 (upper mid Isaac domain-rand)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
    ),
    "fri12_margin01_stiff": ContactConfig(
        name="fri12_margin01_stiff",
        note="friction1.2 + margin0.01 + solref stiff",
        foot_friction=(1.2, 0.005, 0.0001),
        floor_friction=(1.2, 0.005, 0.0001),
        step_friction=(1.2, 0.005, 0.0001),
        margin=0.01,
        solref=(0.01, 1.0),
    ),
    # ---- v2: stability levers on top of fri15 (one at a time) ----
    "fri15_solref_stiff": ContactConfig(
        name="fri15_solref_stiff",
        note="fri15 + solref=(0.01,1) harder contact (closer PhysX)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.01, 1.0),
    ),
    "fri15_solref_stiff005": ContactConfig(
        name="fri15_solref_stiff005",
        note="fri15 + solref=(0.005,1) very stiff",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
    ),
    "fri15_arm0": ContactConfig(
        name="fri15_arm0",
        note="fri15 + armature 0 (Isaac asset.armature=0)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        armature=0.0,
    ),
    "fri15_damp0": ContactConfig(
        name="fri15_damp0",
        note="fri15 + joint damping 0.1→0 (strip MuJoCo extra viscous)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        joint_damping=0.0,
    ),
    "fri15_damp05": ContactConfig(
        name="fri15_damp05",
        note="fri15 + joint damping 0.1→0.5 (extra joint viscous)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        joint_damping=0.5,
    ),
    "fri15_margin01": ContactConfig(
        name="fri15_margin01",
        note="fri15 + margin 0.01 (Isaac physx.contact_offset)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        margin=0.01,
    ),
    "fri15_impratio1": ContactConfig(
        name="fri15_impratio1",
        note="fri15 + impratio 1",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        impratio=1.0,
    ),
    "fri15_solimp_hard": ContactConfig(
        name="fri15_solimp_hard",
        note="fri15 + solimp closer to hard (0.95,0.99,0.001)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    ),
    "fri15_kd15": ContactConfig(
        name="fri15_kd15",
        note="fri15 + Kd 1.0→1.5 (explore; Isaac Kd=1)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        kd=1.5,
    ),
    "fri15_dt002_dec10": ContactConfig(
        name="fri15_dt002_dec10",
        note="fri15 + dt=0.002 decimation=10 (50Hz policy, finer contact)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        timestep=0.002,
        decimation=10,
    ),
    "fri15_floss0": ContactConfig(
        name="fri15_floss0",
        note="fri15 + joint frictionloss 0.2→0",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        joint_frictionloss=0.0,
    ),
    # Likely combo candidates (filled after 1-trial scan)
    "fri15_stiff_arm0": ContactConfig(
        name="fri15_stiff_arm0",
        note="fri15 + solref stiff + armature0",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.01, 1.0),
        armature=0.0,
    ),
    "fri15_stiff_solimp": ContactConfig(
        name="fri15_stiff_solimp",
        note="fri15 + solref stiff + hard solimp",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.01, 1.0),
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    ),
    "fri12_stiff_arm0": ContactConfig(
        name="fri12_stiff_arm0",
        note="friction1.2 + solref stiff + armature0 (closer Isaac mean fri)",
        foot_friction=(1.2, 0.005, 0.0001),
        floor_friction=(1.2, 0.005, 0.0001),
        step_friction=(1.2, 0.005, 0.0001),
        solref=(0.01, 1.0),
        armature=0.0,
    ),
    "fri15_solref_008": ContactConfig(
        name="fri15_solref_008",
        note="fri15 + solref=(0.008,1) mid stiff",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.008, 1.0),
    ),
    "fri15_dt002_stiff005": ContactConfig(
        name="fri15_dt002_stiff005",
        note="fri15 + dt=0.002/dec10 + solref=(0.005,1)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        timestep=0.002,
        decimation=10,
        solref=(0.005, 1.0),
    ),
    "fri15_dt002_stiff": ContactConfig(
        name="fri15_dt002_stiff",
        note="fri15 + dt=0.002/dec10 + solref=(0.01,1)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        timestep=0.002,
        decimation=10,
        solref=(0.01, 1.0),
    ),
    "fri15_dt002_solimp": ContactConfig(
        name="fri15_dt002_solimp",
        note="fri15 + dt=0.002/dec10 + hard solimp",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        timestep=0.002,
        decimation=10,
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    ),
    "stable_v2": ContactConfig(
        name="stable_v2",
        note="v2 default: fri15 + solref=(0.005,1) + dt=0.002/dec10 "
        "(harder contact + finer MuJoCo integrate @ 50Hz policy)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
    ),

    # ---- v3: reduce leg flail on top of stable_v2 (one lever at a time) ----
    "sv2_solimp_hard": ContactConfig(
        name="sv2_solimp_hard",
        note="stable_v2 + hard solimp (0.95,0.99,0.001)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    ),
    "sv2_solimp_mid": ContactConfig(
        name="sv2_solimp_mid",
        note="stable_v2 + mid solimp (0.92,0.97,0.005)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        solimp=(0.92, 0.97, 0.005, 0.5, 2.0),
    ),
    "sv2_damp03": ContactConfig(
        name="sv2_damp03",
        note="stable_v2 + joint damping 0.1→0.3",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.3,
    ),
    "sv2_damp05": ContactConfig(
        name="sv2_damp05",
        note="stable_v2 + joint damping 0.1→0.5",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.5,
    ),
    "sv2_damp10": ContactConfig(
        name="sv2_damp10",
        note="stable_v2 + joint damping 0.1→1.0",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=1.0,
    ),
    "sv2_arm005": ContactConfig(
        name="sv2_arm005",
        note="stable_v2 + armature 0.01→0.005 (toward Isaac 0)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        armature=0.005,
    ),
    "sv2_arm02": ContactConfig(
        name="sv2_arm02",
        note="stable_v2 + armature 0.01→0.02 (more rotor inertia)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        armature=0.02,
    ),
    "sv2_condim3": ContactConfig(
        name="sv2_condim3",
        note="stable_v2 + foot condim 6→3 (drop torsion friction)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        foot_condim=3,
    ),
    "sv2_fri_spin0": ContactConfig(
        name="sv2_fri_spin0",
        note="stable_v2 + near-zero torsional/rolling friction",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
    ),
    "sv2_fri_spinhi": ContactConfig(
        name="sv2_fri_spinhi",
        note="stable_v2 + higher torsional friction (0.02)",
        foot_friction=(1.5, 0.02, 0.001),
        floor_friction=(1.5, 0.02, 0.001),
        step_friction=(1.5, 0.02, 0.001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
    ),
    "sv2_fri12": ContactConfig(
        name="sv2_fri12",
        note="stable_v2 but sliding fri 1.2 (closer Isaac mean)",
        foot_friction=(1.2, 0.005, 0.0001),
        floor_friction=(1.2, 0.005, 0.0001),
        step_friction=(1.2, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
    ),
    "sv2_impratio10": ContactConfig(
        name="sv2_impratio10",
        note="stable_v2 + impratio 100→10",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        impratio=10.0,
    ),
    "sv2_impratio1": ContactConfig(
        name="sv2_impratio1",
        note="stable_v2 + impratio 100→1",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        impratio=1.0,
    ),
    "sv2_pyramid": ContactConfig(
        name="sv2_pyramid",
        note="stable_v2 + pyramidal friction cone",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        cone="pyramidal",
    ),
    "sv2_margin002": ContactConfig(
        name="sv2_margin002",
        note="stable_v2 + margin 0.001→0.002 (careful)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        margin=0.002,
    ),
    "sv2_margin005": ContactConfig(
        name="sv2_margin005",
        note="stable_v2 + margin 0.001→0.005 (careful)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        margin=0.005,
    ),
    "sv2_ema02": ContactConfig(
        name="sv2_ema02",
        note="stable_v2 + action EMA 0.2 (mild rate limit)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        action_ema=0.2,
    ),
    "sv2_ema05": ContactConfig(
        name="sv2_ema05",
        note="stable_v2 + action EMA 0.5",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        action_ema=0.5,
    ),
    "sv2_foot021": ContactConfig(
        name="sv2_foot021",
        note="stable_v2 + foot sphere r 0.022→0.021",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        foot_size=0.021,
    ),
    "sv2_foot024": ContactConfig(
        name="sv2_foot024",
        note="stable_v2 + foot sphere r 0.022→0.024",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        foot_size=0.024,
    ),
    "sv2_torq08": ContactConfig(
        name="sv2_torq08",
        note="stable_v2 + torque clip 0.8×URDF (mild; Isaac soft=0.4 reward-only)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        torque_scale=0.8,
    ),
    "sv2_kd15": ContactConfig(
        name="sv2_kd15",
        note="stable_v2 + Kd 1.0→1.5 (extra PD damp on joints)",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        kd=1.5,
    ),
    "sv2_floss05": ContactConfig(
        name="sv2_floss05",
        note="stable_v2 + joint frictionloss 0.2→0.5",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_frictionloss=0.5,
    ),
    # combo slots filled after 1-trial flail scan
    "sv2_damp03_solimp": ContactConfig(
        name="sv2_damp03_solimp",
        note="stable_v2 + damp0.3 + hard solimp",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.3,
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    ),
    "sv2_damp03_spin0": ContactConfig(
        name="sv2_damp03_spin0",
        note="stable_v2 + damp0.3 + near-zero spin friction",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.3,
    ),
    "sv2_damp05_solimp": ContactConfig(
        name="sv2_damp05_solimp",
        note="stable_v2 + damp0.5 + hard solimp",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.5,
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    ),
    "sv2_damp03_ema02": ContactConfig(
        name="sv2_damp03_ema02",
        note="stable_v2 + damp0.3 + action EMA 0.2",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.3,
        action_ema=0.2,
    ),
    "sv2_damp03_condim3": ContactConfig(
        name="sv2_damp03_condim3",
        note="stable_v2 + damp0.3 + foot condim3",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.3,
        foot_condim=3,
    ),

    "sv2_damp10_condim3": ContactConfig(
        name="sv2_damp10_condim3",
        note="stable_v2 + damp1.0 + foot condim3",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=1.0,
        foot_condim=3,
    ),
    "sv2_damp10_arm005": ContactConfig(
        name="sv2_damp10_arm005",
        note="stable_v2 + damp1.0 + armature 0.005",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=1.0,
        armature=0.005,
    ),
    "sv2_damp03_arm005": ContactConfig(
        name="sv2_damp03_arm005",
        note="stable_v2 + damp0.3 + armature 0.005",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.3,
        armature=0.005,
    ),
    "sv2_kd15_condim3": ContactConfig(
        name="sv2_kd15_condim3",
        note="stable_v2 + Kd1.5 + foot condim3",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        kd=1.5,
        foot_condim=3,
    ),
    "sv2_damp10_spin0": ContactConfig(
        name="sv2_damp10_spin0",
        note="stable_v2 + damp1.0 + near-zero spin friction",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=1.0,
    ),
    "sv2_damp03_kd15": ContactConfig(
        name="sv2_damp03_kd15",
        note="stable_v2 + damp0.3 + Kd1.5",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.3,
        kd=1.5,
    ),


    "sv2_damp03_kd15_condim3": ContactConfig(
        name="sv2_damp03_kd15_condim3",
        note="stable_v2 + damp0.3 + Kd1.5 + foot condim3",
        foot_friction=(1.5, 0.005, 0.0001),
        floor_friction=(1.5, 0.005, 0.0001),
        step_friction=(1.5, 0.005, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.3,
        kd=1.5,
        foot_condim=3,
    ),


    "sv2_damp07_spin0": ContactConfig(
        name="sv2_damp07_spin0",
        note="stable_v2 + damp0.7 + near-zero spin friction",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.7,
    ),


    "sv2_damp10_spinlow": ContactConfig(
        name="sv2_damp10_spinlow",
        note="stable_v2 + damp1.0 + low (not zero) spin fri 0.001",
        foot_friction=(1.5, 0.001, 0.0001),
        floor_friction=(1.5, 0.001, 0.0001),
        step_friction=(1.5, 0.001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=1.0,
    ),
    "sv2_damp10_spin0_m005": ContactConfig(
        name="sv2_damp10_spin0_m005",
        note="stable_v2 + damp1.0 + spin0 + margin0.005",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=1.0,
        margin=0.005,
    ),
    "sv2_damp08_spin0": ContactConfig(
        name="sv2_damp08_spin0",
        note="stable_v2 + damp0.8 + near-zero spin",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.8,
    ),
    "sv2_damp10_spin0_arm005": ContactConfig(
        name="sv2_damp10_spin0_arm005",
        note="stable_v2 + damp1.0 + spin0 + armature0.005",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=1.0,
        armature=0.005,
    ),


    "sv2_damp03_arm005_spin0": ContactConfig(
        name="sv2_damp03_arm005_spin0",
        note="stable_v2 + damp0.3 + armature0.005 + spin0",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.3,
        armature=0.005,
    ),
    "sv2_damp05_spin0": ContactConfig(
        name="sv2_damp05_spin0",
        note="stable_v2 + damp0.5 + spin0",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.5,
    ),
    "sv2_damp06_spin0": ContactConfig(
        name="sv2_damp06_spin0",
        note="stable_v2 + damp0.6 + spin0",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
    ),
    "sv2_damp04_spin0_arm005": ContactConfig(
        name="sv2_damp04_spin0_arm005",
        note="stable_v2 + damp0.4 + spin0 + armature0.005",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.4,
        armature=0.005,
    ),

    "stable_v3": ContactConfig(
        name="stable_v3",
        note="v3 default: stable_v2 + joint_damping=0.6 + near-zero torsional/rolling fri "
        "(reduce leg flail toward Isaac; alias sv2_damp06_spin0)",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
    ),

    # ---- v4: push calf/swing closer to Isaac on top of stable_v3 ----
    "sv3_kd12": ContactConfig(
        name="sv3_kd12",
        note="stable_v3 + Kd 1.0→1.2 (mild PD damp)",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        kd=1.2,
    ),
    "sv3_kd15": ContactConfig(
        name="sv3_kd15",
        note="stable_v3 + Kd 1.0→1.5",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        kd=1.5,
    ),
    "sv3_kd20": ContactConfig(
        name="sv3_kd20",
        note="stable_v3 + Kd 1.0→2.0",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        kd=2.0,
    ),
    "sv3_floss03": ContactConfig(
        name="sv3_floss03",
        note="stable_v3 + joint frictionloss 0.2→0.3",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        joint_frictionloss=0.3,
    ),
    "sv3_floss05": ContactConfig(
        name="sv3_floss05",
        note="stable_v3 + joint frictionloss 0.2→0.5",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        joint_frictionloss=0.5,
    ),
    "sv3_ema01": ContactConfig(
        name="sv3_ema01",
        note="stable_v3 + action EMA 0.1 (very mild)",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        action_ema=0.1,
    ),
    "sv3_ema02": ContactConfig(
        name="sv3_ema02",
        note="stable_v3 + action EMA 0.2",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        action_ema=0.2,
    ),
    "sv3_margin005": ContactConfig(
        name="sv3_margin005",
        note="stable_v3 + margin 0.001→0.005",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        margin=0.005,
    ),
    "sv3_arm015": ContactConfig(
        name="sv3_arm015",
        note="stable_v3 + armature 0.01→0.015",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        armature=0.015,
    ),
    "sv3_calf12": ContactConfig(
        name="sv3_calf12",
        note="stable_v3 + calf_damping=1.2 (hip/thigh stay 0.6)",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=1.2,
    ),
    "sv3_calf18": ContactConfig(
        name="sv3_calf18",
        note="stable_v3 + calf_damping=1.8",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=1.8,
    ),
    "sv3_calf25": ContactConfig(
        name="sv3_calf25",
        note="stable_v3 + calf_damping=2.5",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
    ),
    "sv3_ascale22": ContactConfig(
        name="sv3_ascale22",
        note="stable_v3 + action_scale 0.25→0.22 (mild)",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        action_scale=0.22,
    ),
    "sv3_torq09": ContactConfig(
        name="sv3_torq09",
        note="stable_v3 + torque clip 0.9×",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        torque_scale=0.9,
    ),

    "sv3_calf20": ContactConfig(
        name="sv3_calf20",
        note="stable_v3 + calf_damping=2.0",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.0,
    ),
    "sv3_calf22": ContactConfig(
        name="sv3_calf22",
        note="stable_v3 + calf_damping=2.2",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.2,
    ),
    "sv3_calf30": ContactConfig(
        name="sv3_calf30",
        note="stable_v3 + calf_damping=3.0",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=3.0,
    ),
    "sv3_calf25_floss03": ContactConfig(
        name="sv3_calf25_floss03",
        note="stable_v3 + calf_damping=2.5 + frictionloss 0.3",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        joint_frictionloss=0.3,
    ),

    # placeholder; overwritten after scan with winner alias
    "stable_v4": ContactConfig(
        name="stable_v4",
        note="v4 default: stable_v3 + calf_damping=2.5 (hip/thigh stay 0.6; alias sv3_calf25)",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
    ),
    # ---- front-leg climb A/B (keep plane depth + pass window) ----
    "front_calf35": ContactConfig(
        name="front_calf35",
        note="stable_v4 + front calf damp 3.5",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.5,
    ),
    "front_calf40": ContactConfig(
        name="front_calf40",
        note="stable_v4 + front calf damp 4.0",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=4.0,
    ),
    "front_calf50": ContactConfig(
        name="front_calf50",
        note="stable_v4 + front calf damp 5.0",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=5.0,
    ),
    "front_thigh12": ContactConfig(
        name="front_thigh12",
        note="stable_v4 + front thigh damp 1.2",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_thigh_damping=1.2,
    ),
    "front_calf40_thigh10": ContactConfig(
        name="front_calf40_thigh10",
        note="stable_v4 + front calf 4.0 + thigh 1.0",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=4.0,
        front_thigh_damping=1.0,
    ),
    "front_kd15": ContactConfig(
        name="front_kd15",
        note="stable_v4 + front Kd*1.5",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_kd_scale=1.5,
    ),
    "front_kd20": ContactConfig(
        name="front_kd20",
        note="stable_v4 + front Kd*2.0",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_kd_scale=2.0,
    ),
    "front_ascale22": ContactConfig(
        name="front_ascale22",
        note="stable_v4 + front action_scale 0.22",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_action_scale=0.22,
    ),
    "front_ascale20": ContactConfig(
        name="front_ascale20",
        note="stable_v4 + front action_scale 0.20",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_action_scale=0.20,
    ),
    "front_calf40_kd15": ContactConfig(
        name="front_calf40_kd15",
        note="stable_v4 + front calf 4.0 + Kd*1.5",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=4.0,
        front_kd_scale=1.5,
    ),
    "front_calf40_ascale22": ContactConfig(
        name="front_calf40_ascale22",
        note="stable_v4 + front calf 4.0 + action_scale 0.22",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=4.0,
        front_action_scale=0.22,
    ),
    "front_calf38_thigh10": ContactConfig(
        name="front_calf38_thigh10",
        note="GO2-31 fix: stable_v4 + front calf 3.8 + front thigh 1.0",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
    ),
    "front_calf45_thigh12": ContactConfig(
        name="front_calf45_thigh12",
        note="stable_v4 + front calf 4.5 + thigh 1.2",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=4.5,
        front_thigh_damping=1.2,
    ),

    "isaac_align_base": ContactConfig(
        name="isaac_align_base",
        note="GO2-32 baseline alias = front_calf38_thigh10",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
    ),
    "ia_fc36_th10": ContactConfig(
        name="ia_fc36_th10",
        note="fc38 lineage: front calf 3.6 thigh 1.0",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.6,
        front_thigh_damping=1.0,
    ),
    "ia_fc37_th10": ContactConfig(
        name="ia_fc37_th10",
        note="fc38 lineage: front calf 3.7 thigh 1.0",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.7,
        front_thigh_damping=1.0,
    ),
    "ia_fc39_th10": ContactConfig(
        name="ia_fc39_th10",
        note="fc38 lineage: front calf 3.9 thigh 1.0",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.9,
        front_thigh_damping=1.0,
    ),
    "ia_fc40_th10": ContactConfig(
        name="ia_fc40_th10",
        note="fc38 lineage: front calf 4.0 thigh 1.0",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=4.0,
        front_thigh_damping=1.0,
    ),
    "ia_fc38_th8": ContactConfig(
        name="ia_fc38_th8",
        note="fc38 lineage: front calf 3.8 thigh 0.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=0.8,
    ),
    "ia_fc38_th9": ContactConfig(
        name="ia_fc38_th9",
        note="fc38 lineage: front calf 3.8 thigh 0.9",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=0.9,
    ),
    "ia_fc38_th11": ContactConfig(
        name="ia_fc38_th11",
        note="fc38 lineage: front calf 3.8 thigh 1.1",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.1,
    ),
    "ia_fc36_th8": ContactConfig(
        name="ia_fc36_th8",
        note="fc38 lineage: front calf 3.6 thigh 0.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.6,
        front_thigh_damping=0.8,
    ),
    "ia_fc37_th9": ContactConfig(
        name="ia_fc37_th9",
        note="fc38 lineage: front calf 3.7 thigh 0.9",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.7,
        front_thigh_damping=0.9,
    ),
    "ia_fc39_th11": ContactConfig(
        name="ia_fc39_th11",
        note="fc38 lineage: front calf 3.9 thigh 1.1",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.9,
        front_thigh_damping=1.1,
    ),
    "ia_fhip8": ContactConfig(
        name="ia_fhip8",
        note="fc38 + front hip damp 0.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        front_hip_damping=0.8,
    ),
    "ia_fhip10": ContactConfig(
        name="ia_fhip10",
        note="fc38 + front hip damp 1.0",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        front_hip_damping=1.0,
    ),
    "ia_fhip12": ContactConfig(
        name="ia_fhip12",
        note="fc38 + front hip damp 1.2",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        front_hip_damping=1.2,
    ),
    "ia_rth8": ContactConfig(
        name="ia_rth8",
        note="fc38 + rear thigh 0.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        rear_thigh_damping=0.8,
    ),
    "ia_rth10": ContactConfig(
        name="ia_rth10",
        note="fc38 + rear thigh 1.0",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        rear_thigh_damping=1.0,
    ),
    "ia_rcalf22": ContactConfig(
        name="ia_rcalf22",
        note="fc38 + rear calf 2.2",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        rear_calf_damping=2.2,
    ),
    "ia_rcalf28": ContactConfig(
        name="ia_rcalf28",
        note="fc38 + rear calf 2.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        rear_calf_damping=2.8,
    ),
    "ia_rcalf30": ContactConfig(
        name="ia_rcalf30",
        note="fc38 + rear calf 3.0",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        rear_calf_damping=3.0,
    ),
    "ia_thigh8": ContactConfig(
        name="ia_thigh8",
        note="fc38 + all thigh 0.8 (front stays 1.0)",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        thigh_damping=0.8,
    ),
    "ia_thigh10": ContactConfig(
        name="ia_thigh10",
        note="fc38 + all thigh 1.0 (front stays 1.0)",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        thigh_damping=1.0,
    ),
    "ia_hip8": ContactConfig(
        name="ia_hip8",
        note="fc38 + all hip 0.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "ia_hip10": ContactConfig(
        name="ia_hip10",
        note="fc38 + all hip 1.0",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=1.0,
    ),
    "ia_solref4": ContactConfig(
        name="ia_solref4",
        note="fc38 + solref=(0.004, 1.0)",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.004, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
    ),
    "ia_solref6": ContactConfig(
        name="ia_solref6",
        note="fc38 + solref=(0.006, 1.0)",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.006, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
    ),
    "ia_solref8": ContactConfig(
        name="ia_solref8",
        note="fc38 + solref=(0.008, 1.0)",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.008, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
    ),
    "ia_solref3": ContactConfig(
        name="ia_solref3",
        note="fc38 + solref=(0.003, 1.0)",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.003, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
    ),
    "ia_floss1": ContactConfig(
        name="ia_floss1",
        note="fc38 + frictionloss 0.1",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        joint_frictionloss=0.1,
    ),
    "ia_floss2": ContactConfig(
        name="ia_floss2",
        note="fc38 + frictionloss 0.2",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        joint_frictionloss=0.2,
    ),
    "ia_kd12": ContactConfig(
        name="ia_kd12",
        note="fc38 + Kd=1.2",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        kd=1.2,
    ),
    "ia_arm015": ContactConfig(
        name="ia_arm015",
        note="fc38 + armature 0.015",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        armature=0.015,
    ),
    "ia_margin002": ContactConfig(
        name="ia_margin002",
        note="fc38 + margin 0.002",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        margin=0.002,
    ),
    "ia_fc36_th08_rth08": ContactConfig(
        name="ia_fc36_th08_rth08",
        note="fc38 micro: front 3.6/0.8 + rear thigh 0.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.6,
        front_thigh_damping=0.8,
        rear_thigh_damping=0.8,
    ),
    "ia_fc37_th09_sol006": ContactConfig(
        name="ia_fc37_th09_sol006",
        note="fc38 micro: front 3.7/0.9 + solref 0.006",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.006, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.7,
        front_thigh_damping=0.9,
    ),
    "ia_fc38_rth08_sol006": ContactConfig(
        name="ia_fc38_rth08_sol006",
        note="fc38 + rear thigh 0.8 + solref 0.006",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.006, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        rear_thigh_damping=0.8,
    ),
    "ia_fc38_fhip08_rth08": ContactConfig(
        name="ia_fc38_fhip08_rth08",
        note="fc38 + front hip 0.8 + rear thigh 0.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        front_hip_damping=0.8,
        rear_thigh_damping=0.8,
    ),
    "ia_fc39_th10_rcalf28": ContactConfig(
        name="ia_fc39_th10_rcalf28",
        note="fc38 lineage front 3.9 + rear calf 2.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.9,
        front_thigh_damping=1.0,
        rear_calf_damping=2.8,
    ),
    "ia_fc38_thigh08_all": ContactConfig(
        name="ia_fc38_thigh08_all",
        note="fc38 but all thigh=0.8, front thigh stays 1.0",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        thigh_damping=0.8,
    ),
    "ia_fc36_th10_hip08": ContactConfig(
        name="ia_fc36_th10_hip08",
        note="front calf 3.6 thigh 1.0 + all hip 0.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.6,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "ia_fc38_sol006_floss01": ContactConfig(
        name="ia_fc38_sol006_floss01",
        note="fc38 + solref 0.006 + floss 0.1",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.006, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        joint_frictionloss=0.1,
    ),
    "ia_fc37_th10_rth08_sol006": ContactConfig(
        name="ia_fc37_th10_rth08_sol006",
        note="front 3.7/1.0 + rear thigh 0.8 + solref 0.006",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.006, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.7,
        front_thigh_damping=1.0,
        rear_thigh_damping=0.8,
    ),
    "ia_fhip10_sol004": ContactConfig(
        name="ia_fhip10_sol004",
        note="fc38 + front hip 1.0 + solref 0.004",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.004, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        front_hip_damping=1.0,
    ),
    "ia_fhip10_sol006": ContactConfig(
        name="ia_fhip10_sol006",
        note="fc38 + front hip 1.0 + solref 0.006",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.006, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        front_hip_damping=1.0,
    ),
    "ia_hip8_sol004": ContactConfig(
        name="ia_hip8_sol004",
        note="fc38 + all hip 0.8 + solref 0.004",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.004, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "ia_fhip10_hip08": ContactConfig(
        name="ia_fhip10_hip08",
        note="fc38 + front hip 1.0 overrides all-hip 0.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        front_hip_damping=1.0,
    ),
    "ia_fhip10_rcalf28": ContactConfig(
        name="ia_fhip10_rcalf28",
        note="fc38 + front hip 1.0 + rear calf 2.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        front_hip_damping=1.0,
        rear_calf_damping=2.8,
    ),
    "ia_sol004_hip08": ContactConfig(
        name="ia_sol004_hip08",
        note="fc38 + solref 0.004 + all hip 0.8",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.004, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "ia_fhip10_kd12": ContactConfig(
        name="ia_fhip10_kd12",
        note="fc38 + front hip 1.0 + Kd 1.2",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        front_hip_damping=1.0,
        kd=1.2,
    ),
    "ia_sol004_kd12": ContactConfig(
        name="ia_sol004_kd12",
        note="fc38 + solref 0.004 + Kd 1.2",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.004, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        kd=1.2,
    ),
    "ia_fc37_th10_fhip10": ContactConfig(
        name="ia_fc37_th10_fhip10",
        note="front calf 3.7 thigh 1.0 + front hip 1.0",
        foot_friction=(1.5, 0.0001, 0.0001),
        floor_friction=(1.5, 0.0001, 0.0001),
        step_friction=(1.5, 0.0001, 0.0001),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.7,
        front_thigh_damping=1.0,
        front_hip_damping=1.0,
    ),
    "stable_isaac_v5": ContactConfig(
        name="stable_isaac_v5",
        note="GO2-32 winner: front_calf38_thigh10 + all-hip damping 0.8 (alias ia_hip8)",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "damping_isaac": ContactConfig(
        name="damping_isaac",
        note="alias of stable_isaac_v5 / ia_hip8",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "damping_isaac": ContactConfig(
        name="damping_isaac",
        note="alias of stable_isaac_v5 / ia_hip8",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    # ---- GO2-33 v6: single-lever deltas ON stable_isaac_v5 (hip8) ----
    "v6_base": ContactConfig(
        name="v6_base",
        note="GO2-33 baseline = stable_isaac_v5 / ia_hip8",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_hip70": ContactConfig(
        name="v6_hip70",
        note="hip8→ all hip 0.70",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.70,
    ),
    "v6_hip75": ContactConfig(
        name="v6_hip75",
        note="hip8→ all hip 0.75",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.75,
    ),
    "v6_hip85": ContactConfig(
        name="v6_hip85",
        note="hip8→ all hip 0.85",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.85,
    ),
    "v6_hip90": ContactConfig(
        name="v6_hip90",
        note="hip8→ all hip 0.90",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.90,
    ),
    "v6_fc36": ContactConfig(
        name="v6_fc36",
        note="hip8 + front calf 3.6",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.6,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_fc37": ContactConfig(
        name="v6_fc37",
        note="hip8 + front calf 3.7",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.7,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_fc39": ContactConfig(
        name="v6_fc39",
        note="hip8 + front calf 3.9",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.9,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_fc40": ContactConfig(
        name="v6_fc40",
        note="hip8 + front calf 4.0",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=4.0,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_fth09": ContactConfig(
        name="v6_fth09",
        note="hip8 + front thigh 0.9",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=0.9,
        hip_damping=0.8,
    ),
    "v6_fth11": ContactConfig(
        name="v6_fth11",
        note="hip8 + front thigh 1.1",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.1,
        hip_damping=0.8,
    ),
    "v6_rc22": ContactConfig(
        name="v6_rc22",
        note="hip8 + rear calf 2.2",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        rear_calf_damping=2.2,
    ),
    "v6_rc26": ContactConfig(
        name="v6_rc26",
        note="hip8 + rear calf 2.6",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        rear_calf_damping=2.6,
    ),
    "v6_rc28": ContactConfig(
        name="v6_rc28",
        note="hip8 + rear calf 2.8",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        rear_calf_damping=2.8,
    ),
    "v6_sol45": ContactConfig(
        name="v6_sol45",
        note="hip8 + solref timeconst 0.0045 (micro; 0.004 broke)",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.0045, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_sol48": ContactConfig(
        name="v6_sol48",
        note="hip8 + solref timeconst 0.0048",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.0048, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_sol_d12": ContactConfig(
        name="v6_sol_d12",
        note="hip8 + solref dampratio 1.2",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.2),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_sol_d09": ContactConfig(
        name="v6_sol_d09",
        note="hip8 + solref dampratio 0.9",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 0.9),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_floss05": ContactConfig(
        name="v6_floss05",
        note="hip8 + frictionloss 0.05",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        joint_frictionloss=0.05,
    ),
    "v6_floss10": ContactConfig(
        name="v6_floss10",
        note="hip8 + frictionloss 0.10",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        joint_frictionloss=0.10,
    ),
    "v6_kd105": ContactConfig(
        name="v6_kd105",
        note="hip8 + Kd=1.05",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        kd=1.05,
    ),
    "v6_kd11": ContactConfig(
        name="v6_kd11",
        note="hip8 + Kd=1.1",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        kd=1.1,
    ),
    "v6_arm012": ContactConfig(
        name="v6_arm012",
        note="hip8 + armature 0.012",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        armature=0.012,
    ),
    "v6_ema95": ContactConfig(
        name="v6_ema95",
        note="hip8 + action_ema 0.95 (mild smooth)",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        action_ema=0.95,
    ),
    "v6_ema90": ContactConfig(
        name="v6_ema90",
        note="hip8 + action_ema 0.90",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        action_ema=0.90,
    ),
    "v6_rhip9": ContactConfig(
        name="v6_rhip9",
        note="hip8 + rear hip override 0.9 (front stays 0.8)",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        rear_hip_damping=0.9,
    ),
    "v6_fhip75": ContactConfig(
        name="v6_fhip75",
        note="hip8 + front hip override 0.75",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        front_hip_damping=0.75,
    ),
    "v6_fhip85": ContactConfig(
        name="v6_fhip85",
        note="hip8 + front hip override 0.85",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
        front_hip_damping=0.85,
    ),

    "v6_sol_d092": ContactConfig(
        name="v6_sol_d092",
        note="hip8 + solref dampratio 0.92",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 0.92),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_sol_d095": ContactConfig(
        name="v6_sol_d095",
        note="hip8 + solref dampratio 0.95",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 0.95),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_sol_d098": ContactConfig(
        name="v6_sol_d098",
        note="hip8 + solref dampratio 0.98",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 0.98),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_sol42": ContactConfig(
        name="v6_sol42",
        note="hip8 + solref timeconst 0.0042",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.0042, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_sol46": ContactConfig(
        name="v6_sol46",
        note="hip8 + solref timeconst 0.0046",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.0046, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "v6_sol47": ContactConfig(
        name="v6_sol47",
        note="hip8 + solref timeconst 0.0047",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.0047, 1.0),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),

    "stable_isaac_v6": ContactConfig(
        name="stable_isaac_v6",
        note="GO2-33 winner: stable_isaac_v5 (hip8) + solref dampratio 0.9 (alias v6_sol_d09)",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 0.9),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),
    "damping_isaac_v6": ContactConfig(
        name="damping_isaac_v6",
        note="alias of stable_isaac_v6 / v6_sol_d09",
        foot_friction=(1.5, 1e-4, 1e-4),
        floor_friction=(1.5, 1e-4, 1e-4),
        step_friction=(1.5, 1e-4, 1e-4),
        solref=(0.005, 0.9),
        timestep=0.002,
        decimation=10,
        joint_damping=0.6,
        calf_damping=2.5,
        front_calf_damping=3.8,
        front_thigh_damping=1.0,
        hip_damping=0.8,
    ),

}


def apply_contact_config(model: mujoco.MjModel, cfg: ContactConfig) -> None:
    """Mutate model in-place according to one contact config."""
    if cfg.timestep is not None:
        model.opt.timestep = cfg.timestep
    if cfg.impratio is not None:
        model.opt.impratio = cfg.impratio

    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOM_NAMES]
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    step_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "step40")

    def set_friction(gid: int, fri: Tuple[float, float, float]) -> None:
        if gid < 0:
            return
        model.geom_friction[gid] = fri

    if cfg.foot_friction is not None:
        for gid in foot_ids:
            set_friction(gid, cfg.foot_friction)
    if cfg.floor_friction is not None and floor_id >= 0:
        set_friction(floor_id, cfg.floor_friction)
    if cfg.step_friction is not None and step_id >= 0:
        set_friction(step_id, cfg.step_friction)

    # Freejoint occupies dof 0..5; only touch actuated hinge dofs.
    hinge_dofs = slice(6, None)

    if cfg.armature is not None:
        model.dof_armature[hinge_dofs] = cfg.armature

    if cfg.margin is not None:
        # Apply to feet / floor / step (contact-relevant)
        for gid in foot_ids + [floor_id, step_id]:
            if gid >= 0:
                model.geom_margin[gid] = cfg.margin

    if cfg.solref is not None:
        for gid in foot_ids + [floor_id, step_id]:
            if gid >= 0:
                model.geom_solref[gid] = cfg.solref

    if cfg.solimp is not None:
        for gid in foot_ids + [floor_id, step_id]:
            if gid >= 0:
                model.geom_solimp[gid] = cfg.solimp

    if cfg.joint_frictionloss is not None:
        model.dof_frictionloss[hinge_dofs] = cfg.joint_frictionloss
    if cfg.joint_damping is not None:
        model.dof_damping[hinge_dofs] = cfg.joint_damping

    if cfg.calf_damping is not None:
        # Knee/calf hinges only — leave hip/thigh at joint_damping (or XML)
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if "calf" not in name.lower():
                continue
            adr = int(model.jnt_dofadr[j])
            if adr >= 0:
                model.dof_damping[adr] = float(cfg.calf_damping)

    # Front-leg-only damping overrides (FR_/FL_ joints)
    if (
        cfg.front_calf_damping is not None
        or cfg.front_thigh_damping is not None
        or cfg.front_hip_damping is not None
    ):
        for j in range(model.njnt):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").lower()
            adr = int(model.jnt_dofadr[j])
            if adr < 6:
                continue
            is_front = name.startswith("fl_") or name.startswith("fr_")
            if not is_front:
                continue
            if "calf" in name and cfg.front_calf_damping is not None:
                model.dof_damping[adr] = float(cfg.front_calf_damping)
            if "thigh" in name and cfg.front_thigh_damping is not None:
                model.dof_damping[adr] = float(cfg.front_thigh_damping)
            if "hip" in name and cfg.front_hip_damping is not None:
                model.dof_damping[adr] = float(cfg.front_hip_damping)

    # All-thigh / all-hip viscous
    if cfg.thigh_damping is not None:
        for j in range(model.njnt):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").lower()
            adr = int(model.jnt_dofadr[j])
            if adr < 6:
                continue
            if "thigh" in name:
                model.dof_damping[adr] = float(cfg.thigh_damping)
    if cfg.hip_damping is not None:
        for j in range(model.njnt):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").lower()
            adr = int(model.jnt_dofadr[j])
            if adr < 6:
                continue
            if "hip" in name:
                model.dof_damping[adr] = float(cfg.hip_damping)

    # Rear-leg-only overrides (RR_/RL_)
    if (
        cfg.rear_calf_damping is not None
        or cfg.rear_thigh_damping is not None
        or cfg.rear_hip_damping is not None
    ):
        for j in range(model.njnt):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").lower()
            adr = int(model.jnt_dofadr[j])
            if adr < 6:
                continue
            is_rear = name.startswith("rl_") or name.startswith("rr_")
            if not is_rear:
                continue
            if "calf" in name and cfg.rear_calf_damping is not None:
                model.dof_damping[adr] = float(cfg.rear_calf_damping)
            if "thigh" in name and cfg.rear_thigh_damping is not None:
                model.dof_damping[adr] = float(cfg.rear_thigh_damping)
            if "hip" in name and cfg.rear_hip_damping is not None:
                model.dof_damping[adr] = float(cfg.rear_hip_damping)

    # Re-apply front overrides last so they win over thigh_damping/hip_damping
    if (
        cfg.front_calf_damping is not None
        or cfg.front_thigh_damping is not None
        or cfg.front_hip_damping is not None
    ):
        for j in range(model.njnt):
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").lower()
            adr = int(model.jnt_dofadr[j])
            if adr < 6:
                continue
            is_front = name.startswith("fl_") or name.startswith("fr_")
            if not is_front:
                continue
            if "calf" in name and cfg.front_calf_damping is not None:
                model.dof_damping[adr] = float(cfg.front_calf_damping)
            if "thigh" in name and cfg.front_thigh_damping is not None:
                model.dof_damping[adr] = float(cfg.front_thigh_damping)
            if "hip" in name and cfg.front_hip_damping is not None:
                model.dof_damping[adr] = float(cfg.front_hip_damping)

    if cfg.foot_condim is not None:
        for gid in foot_ids:
            if gid >= 0:
                model.geom_condim[gid] = int(cfg.foot_condim)

    if cfg.cone is not None:
        # mujoco.mjtCone: mjCONE_PYRAMIDAL=0, mjCONE_ELLIPTIC=1
        if cfg.cone.lower().startswith("pyr"):
            model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        else:
            model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    if cfg.foot_size is not None:
        for gid in foot_ids:
            if gid >= 0:
                # sphere: size[0] = radius
                model.geom_size[gid][0] = float(cfg.foot_size)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def quat_wxyz_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """MuJoCo quat is (w,x,y,z). Return roll, pitch, yaw."""
    w, x, y, z = [float(v) for v in q]
    # roll (x)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    # pitch (y)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # yaw (z)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def body_rot_mat(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return data.xmat[body_id].reshape(3, 3).copy()


# ---------------------------------------------------------------------------
# Plane-depth mj_ray camera
# ---------------------------------------------------------------------------

class PlaneDepthCamera:
    def __init__(self, model: mujoco.MjModel, pitch_deg: float = CAM_PITCH_DEG):
        self.model = model
        self.pitch_deg = pitch_deg
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self.base_id < 0:
            self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_id < 0:
            raise RuntimeError("base / base_link body not found")

        w, h = DEPTH_ORIGIN
        hfov = math.radians(HFOV_DEG)
        fx = 0.5 * w / math.tan(0.5 * hfov)
        fy = fx  # square pixels
        cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)
        # Pixel rays in camera frame (MuJoCo camera looks along -Z, Y up)
        us = np.arange(w, dtype=np.float64)
        vs = np.arange(h, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)
        x = (uu - cx) / fx
        y = -(vv - cy) / fy  # image v down → cam y up
        z = -np.ones_like(x)
        dirs = np.stack([x, y, z], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        self.dirs_cam = dirs.reshape(-1, 3)  # (H*W, 3)
        self.hw = (h, w)

        # Docs: geom groups 2 and 3 off (skip robot visual+collision self-hits).
        # Floor/step are group 0 and remain visible to rays.
        self.geomgroup = np.ones(6, dtype=np.uint8)
        self.geomgroup[2] = 0
        self.geomgroup[3] = 0

    def camera_pose(self, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
        """Return cam_pos_world (3,), R_cam_to_world (3,3)."""
        R_body = body_rot_mat(data, self.base_id)
        p_body = data.xpos[self.base_id].copy()
        # Body +X forward, +Y left, +Z up. Pitch-down optical axis in body:
        pitch = math.radians(self.pitch_deg)
        forward = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
        # Camera right = robot right = −body Y; look along −Z_cam (= forward).
        right = np.array([0.0, -1.0, 0.0])
        z_cam = -forward
        up = np.cross(z_cam, right)
        up /= np.linalg.norm(up)
        R_cam_body = np.column_stack([right, up, z_cam])
        R_cam_world = R_body @ R_cam_body
        p_cam = p_body + R_body @ CAM_POS_BODY
        return p_cam, R_cam_world

    def render_plane_depth(self, data: mujoco.MjData) -> np.ndarray:
        """Return HxW positive plane depth in meters (far fill = FAR_CLIP)."""
        h, w = self.hw
        nray = h * w
        p_cam, R_cw = self.camera_pose(data)
        dirs_w = np.ascontiguousarray((R_cw @ self.dirs_cam.T).T.reshape(-1), dtype=np.float64)
        plane_scale = -self.dirs_cam[:, 2]  # look along -Z → plane depth = dist * (-dir_z)

        dist = np.full(nray, -1.0, dtype=np.float64)
        geomid = np.full(nray, -1, dtype=np.int32)
        pnt = np.ascontiguousarray(p_cam.reshape(3, 1), dtype=np.float64)
        geomgroup = np.ascontiguousarray(self.geomgroup, dtype=np.uint8)
        mujoco.mj_multiRay(
            self.model,
            data,
            pnt,
            dirs_w,
            geomgroup,
            1,
            self.base_id,
            geomid,
            dist,
            nray,
            FAR_CLIP * 2.0,
        )
        depth = np.full(nray, FAR_CLIP, dtype=np.float64)
        hit = dist >= 0.0
        plane = dist[hit] * plane_scale[hit]
        depth[hit] = np.minimum(plane, FAR_CLIP)
        return depth.reshape(h, w)

    def preprocess(self, depth_m: np.ndarray) -> np.ndarray:
        """Isaac process_depth_image on positive metric depth.

        MuJoCo returns meters in front of the camera. Isaac IMAGE_DEPTH is the
        same distance with a minus sign. Crop, clip to [-far, -near], bicubic
        resize, then negate and map with (d - near) / (far - near) - 0.5.
        """
        # positive meters → Isaac's negative depth
        d = -depth_m.astype(np.float64)
        # crop [:-2, 4:-4] → (58, 98) from (60, 106)
        d = d[:-2, 4:-4]
        d = np.clip(d, -FAR_CLIP, -NEAR_CLIP)
        # OpenCV bicubic to 87x58 (W,H) — d is HxW. Same order as Isaac:
        # clip the negative image, then resize, then normalize.
        d_resized = cv2.resize(
            d, DEPTH_RESIZED, interpolation=cv2.INTER_CUBIC
        )  # (58, 87)
        d_pos = -d_resized
        norm = (d_pos - NEAR_CLIP) / (FAR_CLIP - NEAR_CLIP) - 0.5
        return norm.astype(np.float32)

    def render_depth(self, data: mujoco.MjData) -> np.ndarray:
        """Positive plane-depth in meters. Plane rays stay available via render_plane_depth."""
        return self.render_plane_depth(data)


class RenderedDepthCamera(PlaneDepthCamera):
    """Offscreen MuJoCo depth at the same mount as the plane-ray camera.

    preprocess() sign-flips MuJoCo's positive depth into Isaac IMAGE_DEPTH,
    clips to [-2, 0], bicubic-resizes, then negates and normalizes.

    Resolution 106x60, horizontal FOV 87 deg, body mount [0.24, -0.0175, 0.12].
    Pitch defaults to the midpoint of the Go2 training range and stays fixed
    for the episode. attach_camera / play.py also sample once, not every step.
    The rasterizer includes the robot; plane rays skip geom groups 2 and 3.
    """

    def __init__(self, model: mujoco.MjModel, pitch_deg: float = CAM_PITCH_DEG):
        super().__init__(model, pitch_deg)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), DEPTH_ORIGIN[0])
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), DEPTH_ORIGIN[1])
        self.renderer = mujoco.Renderer(
            model, height=DEPTH_ORIGIN[1], width=DEPTH_ORIGIN[0]
        )
        self.renderer.enable_depth_rendering()
        # vertical half-angle from horizontal FOV and the 106x60 viewport
        self._half_tan_vfov = math.tan(math.radians(HFOV_DEG) * 0.5) * (
            DEPTH_ORIGIN[1] / float(DEPTH_ORIGIN[0])
        )

    def render_depth(self, data: mujoco.MjData) -> np.ndarray:
        # Groups 2 and 3 are the robot visual and collision meshes. The
        # mount sits inside the head mesh, so those geoms fill the image
        # with a ~5 cm self-hit and the student falls before the step.
        # Terrain is group 0 and stays visible. Same mask as the plane rays.
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1
        opt.geomgroup[2] = 0
        opt.geomgroup[3] = 0
        self.renderer.update_scene(data, scene_option=opt)
        p_cam, r_cw = self.camera_pose(data)
        forward = -r_cw[:, 2]
        up = r_cw[:, 1]
        for eye in range(2):
            cam = self.renderer.scene.camera[eye]
            cam.pos[:] = p_cam
            cam.forward[:] = forward
            cam.up[:] = up
            near = float(cam.frustum_near)
            top = self._half_tan_vfov * near
            cam.frustum_top = top
            cam.frustum_bottom = -top
        return np.array(self.renderer.render(), copy=True)

    def close(self) -> None:
        self.renderer.close()


# ---------------------------------------------------------------------------
# Policy stack
# ---------------------------------------------------------------------------

class EPStudentPolicy:
    def __init__(self, vision_path: Path, jit_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.encoder = RecurrentDepthBackbone(n_proprio=N_PROPRIO).to(self.device)
        blob = torch.load(str(vision_path), map_location=self.device, weights_only=False)
        if isinstance(blob, dict) and "depth_encoder_state_dict" in blob:
            sd = blob["depth_encoder_state_dict"]
        elif isinstance(blob, dict) and "model_state_dict" in blob:
            sd = blob["model_state_dict"]
        else:
            sd = blob
        missing, unexpected = self.encoder.load_state_dict(sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"vision load mismatch missing={missing} unexpected={unexpected}")
        self.encoder.eval()
        self.jit = torch.jit.load(str(jit_path), map_location=self.device)
        self.jit.eval()
        self.hidden_reset()
        self.latent = torch.zeros(1, 32, device=self.device)
        self.yaw = torch.zeros(1, 2, device=self.device)

    def hidden_reset(self) -> None:
        self.encoder.reset_hidden()

    @torch.no_grad()
    def update_depth(self, depth_58x87: np.ndarray, proprio53: np.ndarray) -> float:
        """Push GRU one step. proprio53 must have yaw slots 6:8 cleared."""
        d = torch.from_numpy(depth_58x87).to(self.device).unsqueeze(0)  # B,H,W
        p = torch.from_numpy(proprio53.astype(np.float32)).to(self.device).unsqueeze(0).clone()
        p[:, 6:8] = 0.0
        out = self.encoder(d, p)  # B,34
        self.latent = out[:, :32]
        self.yaw = out[:, 32:34]
        return float(torch.linalg.vector_norm(self.latent).item())

    @torch.no_grad()
    def act(self, obs753: np.ndarray) -> np.ndarray:
        o = torch.from_numpy(obs753.astype(np.float32)).to(self.device).unsqueeze(0)
        # write yaw * 1.5 into slots 6:8 (deploy contract)
        o = o.clone()
        o[:, 6:8] = 1.5 * self.yaw
        a = self.jit(o, self.latent)
        if isinstance(a, (tuple, list)):
            a = a[0]
        a = a.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return np.clip(a, -ACTION_CLIP, ACTION_CLIP)


# ---------------------------------------------------------------------------
# Simulator episode
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    cmd: float
    config: str
    trial: int
    passed: int
    max_x: float
    max_z: float
    last_x: float
    last_z: float
    best_streak: int
    on_deck: int
    saturated: int
    first_latent_norm: float
    first_action: List[float]
    depth_updates: int
    steps: int
    note: str = ""
    # Stability metrics (v2)
    roll_rms: float = float("nan")
    pitch_rms: float = float("nan")
    roll_rms_on: float = float("nan")
    pitch_rms_on: float = float("nan")
    angvel_rms: float = float("nan")
    y_rms: float = float("nan")
    max_abs_y: float = float("nan")
    z_std_on: float = float("nan")
    first_on_step: int = -1
    post_climb_hold: int = 0  # consecutive on_deck after first_on, until break
    post_climb_on_total: int = 0  # total on_deck steps after first_on
    fell_after_climb: int = 0  # z collapsed after having been on deck
    end_on_deck: int = 0
    # Flail metrics (v3)
    joint_vel_rms: float = float("nan")
    joint_vel_peak: float = float("nan")
    calf_vel_rms: float = float("nan")
    action_rate_rms: float = float("nan")
    swing_clear_mean: float = float("nan")
    swing_clear_p95: float = float("nan")
    swing_clear_max: float = float("nan")
    stance_frac: float = float("nan")
    # Front-leg climb diagnosis (phase × leg)
    approach_front_calf_rms: float = float("nan")
    approach_rear_calf_rms: float = float("nan")
    approach_front_swing_p95: float = float("nan")
    approach_rear_swing_p95: float = float("nan")
    approach_front_jvel_rms: float = float("nan")
    approach_rear_jvel_rms: float = float("nan")
    approach_pitch_rms: float = float("nan")
    preclimb_front_calf_rms: float = float("nan")
    preclimb_rear_calf_rms: float = float("nan")
    preclimb_front_swing_p95: float = float("nan")
    preclimb_rear_swing_p95: float = float("nan")
    preclimb_front_jvel_rms: float = float("nan")
    preclimb_rear_jvel_rms: float = float("nan")
    preclimb_pitch_rms: float = float("nan")
    preclimb_front_action_rate_rms: float = float("nan")
    climb_front_calf_rms: float = float("nan")
    climb_rear_calf_rms: float = float("nan")
    climb_front_swing_p95: float = float("nan")
    ondeck_front_calf_rms: float = float("nan")
    ondeck_rear_calf_rms: float = float("nan")
    ondeck_front_swing_p95: float = float("nan")
    ondeck_pitch_rms: float = float("nan")
    front_calf_rms: float = float("nan")
    rear_calf_rms: float = float("nan")
    front_swing_p95: float = float("nan")
    rear_swing_p95: float = float("nan")


def foot_contacts(model: mujoco.MjModel, data: mujoco.MjData, foot_geom_ids: Sequence[int]) -> np.ndarray:
    """Return ±0.5 contact flags for FR,FL,RR,RL."""
    flags = np.full(4, -0.5, dtype=np.float64)
    for i in range(data.ncon):
        c = data.contact[i]
        for fi, gid in enumerate(foot_geom_ids):
            if c.geom1 == gid or c.geom2 == gid:
                flags[fi] = 0.5
    return flags


def build_proprio(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    cmd_x: float,
    last_action: np.ndarray,
    contacts: np.ndarray,
    base_id: int,
) -> np.ndarray:
    proprio = np.zeros(N_PROPRIO, dtype=np.float64)
    # ang vel in body frame
    # MuJoCo cvel is 6D spatial velocity in body/world mixed; use sensordata gyro if present
    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    if gyro_id >= 0:
        adr = model.sensor_adr[gyro_id]
        ang = data.sensordata[adr : adr + 3].copy()
    else:
        # fallback: transform subtree angular vel
        ang = data.cvel[base_id][3:6].copy()
    proprio[0:3] = ang * 0.25

    quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
    if quat_id >= 0:
        adr = model.sensor_adr[quat_id]
        q = data.sensordata[adr : adr + 4].copy()
    else:
        q = data.qpos[3:7].copy()
    roll, pitch, yaw = quat_wxyz_to_euler(q)
    proprio[3] = roll
    proprio[4] = pitch
    # 5,6,7 yaw slots leave 0 here (encoder fills 6:8 later on obs copy)
    proprio[10] = cmd_x
    proprio[11] = 1.0  # parkour
    proprio[12] = 0.0

    # Joints in policy/actuator order FR,FL,RR,RL (not raw qpos tree order)
    qj = data.qpos[QPOS_IDX].copy()
    dq = data.qvel[QVEL_IDX].copy()
    proprio[13:25] = (qj - DEFAULT_DOF) * 1.0
    proprio[25:37] = dq * 0.05
    proprio[37:49] = last_action
    proprio[49:53] = contacts
    return proprio


def assemble_obs(proprio: np.ndarray, hist: np.ndarray) -> np.ndarray:
    obs = np.zeros(N_OBS, dtype=np.float64)
    obs[:N_PROPRIO] = proprio
    # scan/priv zeros
    hist_off = N_PROPRIO + N_SCAN + N_PRIV_E + N_PRIV_L
    obs[hist_off:] = hist
    return np.clip(obs, -100.0, 100.0)


def pd_torques(
    q: np.ndarray,
    dq: np.ndarray,
    target: np.ndarray,
    kp: float = KP,
    kd: float = KD,
) -> np.ndarray:
    tau = kp * (target - q) - kd * dq
    return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)


def _rms(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    a = np.asarray(xs, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a)))


def terrain_height_xy(x: float, y: float) -> float:
    """scene_step40 box: center (2.2,0,0.20), half (1.2,0.9,0.20) → top z=0.40."""
    if 1.0 <= x <= 3.4 and -0.9 <= y <= 0.9:
        return 0.40
    return 0.0


def run_episode(
    model: mujoco.MjModel,
    policy: EPStudentPolicy,
    cmd_x: float,
    cfg_name: str,
    trial: int,
    n_steps: int = 700,
    record_path: Optional[Path] = None,
    seed_qpos: Optional[np.ndarray] = None,
    cfg: Optional[ContactConfig] = None,
    depth_backend: str = "render",
    trace: Optional[List[dict]] = None,
) -> EpisodeResult:
    kp = float(cfg.kp) if cfg is not None and cfg.kp is not None else KP
    kd = float(cfg.kd) if cfg is not None and cfg.kd is not None else KD
    decimation = int(cfg.decimation) if cfg is not None and cfg.decimation is not None else DECIMATION
    action_scale = (
        float(cfg.action_scale) if cfg is not None and cfg.action_scale is not None else ACTION_SCALE
    )
    action_ema = (
        float(cfg.action_ema) if cfg is not None and cfg.action_ema is not None else None
    )
    torque_scale = (
        float(cfg.torque_scale) if cfg is not None and cfg.torque_scale is not None else 1.0
    )
    torque_lim = TORQUE_LIMIT * torque_scale
    front_action_scale = (
        float(cfg.front_action_scale)
        if cfg is not None and cfg.front_action_scale is not None
        else None
    )
    front_kd_scale = (
        float(cfg.front_kd_scale)
        if cfg is not None and cfg.front_kd_scale is not None
        else None
    )
    front_kd = kd * front_kd_scale if front_kd_scale is not None else kd

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if seed_qpos is not None:
        data.qpos[:] = seed_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    else:
        # Spawn slightly above ground with policy default pose, then PD-settle.
        data.qpos[0:3] = np.array([0.0, 0.0, 0.35])
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        data.qpos[QPOS_IDX] = DEFAULT_DOF
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        settle_n = max(1, int(round(1.0 / float(model.opt.timestep))))
        for _ in range(settle_n):
            qj = data.qpos[QPOS_IDX].copy()
            dq = data.qvel[QVEL_IDX].copy()
            tau = kp * (DEFAULT_DOF - qj) - kd * dq
            data.ctrl[:] = np.clip(tau, -torque_lim, torque_lim)
            mujoco.mj_step(model, data)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_id < 0:
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    foot_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOM_NAMES
    ]

    if depth_backend == "plane":
        cam = PlaneDepthCamera(model)
    elif depth_backend == "render":
        cam = RenderedDepthCamera(model)
    else:
        raise ValueError(f"unknown depth_backend {depth_backend}")
    policy.hidden_reset()
    policy.latent = torch.zeros(1, 32, device=policy.device)
    policy.yaw = torch.zeros(1, 2, device=policy.device)

    last_action = np.zeros(12, dtype=np.float64)
    action_queue: List[np.ndarray] = [np.zeros(12, dtype=np.float64)] * ACTION_DELAY
    hist = np.zeros(HIST_LEN * N_PROPRIO, dtype=np.float64)
    hist_ready = False

    max_x = -1e9
    max_z = -1e9
    best_streak = 0
    streak = 0
    on_deck = 0
    saturated = 0
    first_latent = float("nan")
    first_action: List[float] = []
    depth_updates = 0

    rolls: List[float] = []
    pitches: List[float] = []
    rolls_on: List[float] = []
    pitches_on: List[float] = []
    angvels: List[float] = []
    ys: List[float] = []
    zs_on: List[float] = []
    first_on_step = -1
    post_climb_hold = 0
    post_climb_hold_cur = 0
    post_climb_on_total = 0
    climbed = False
    fell_after_climb = 0
    hold_broken = False

    joint_vel_norms: List[float] = []
    joint_vel_peaks: List[float] = []
    calf_vel_abs: List[float] = []
    action_rates: List[float] = []
    swing_clears: List[float] = []
    stance_hits = 0
    stance_tot = 0
    prev_action_for_rate = np.zeros(12, dtype=np.float64)
    smoothed_action = np.zeros(12, dtype=np.float64)
    # phase × leg collectors
    ph_front_calf = {k: [] for k in ("approach", "preclimb", "climb", "ondeck")}
    ph_rear_calf = {k: [] for k in ("approach", "preclimb", "climb", "ondeck")}
    ph_front_jvel = {k: [] for k in ("approach", "preclimb", "climb", "ondeck")}
    ph_rear_jvel = {k: [] for k in ("approach", "preclimb", "climb", "ondeck")}
    ph_front_swing = {k: [] for k in ("approach", "preclimb", "climb", "ondeck")}
    ph_rear_swing = {k: [] for k in ("approach", "preclimb", "climb", "ondeck")}
    ph_pitch = {k: [] for k in ("approach", "preclimb", "climb", "ondeck")}
    ph_front_arate = {k: [] for k in ("approach", "preclimb", "climb", "ondeck")}
    all_front_calf: List[float] = []
    all_rear_calf: List[float] = []
    all_front_swing: List[float] = []
    all_rear_swing: List[float] = []

    renderer = None
    frames = []
    if record_path is not None:
        # Expand offscreen framebuffer if model XML defaults are tiny (640).
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 1280)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), 720)
        renderer = mujoco.Renderer(model, height=720, width=1280)

    applied_action = np.zeros(12, dtype=np.float64)

    for step in range(n_steps):
        # --- observe ---
        contacts = foot_contacts(model, data, foot_ids)
        proprio = build_proprio(data, model, cmd_x, last_action, contacts, base_id)
        hist_frame = proprio.copy()
        hist_frame[6:8] = 0.0
        if not hist_ready:
            for h in range(HIST_LEN):
                hist[h * N_PROPRIO : (h + 1) * N_PROPRIO] = hist_frame
            hist_ready = True
        else:
            hist[:-N_PROPRIO] = hist[N_PROPRIO:]
            hist[-N_PROPRIO:] = hist_frame

        # depth every UPDATE_INTERVAL policy steps (and step 0)
        if step % UPDATE_INTERVAL == 0:
            depth_m = cam.render_depth(data)
            depth_n = cam.preprocess(depth_m)
            ln = policy.update_depth(depth_n, proprio.astype(np.float32))
            depth_updates += 1
            if step == 0:
                first_latent = ln

        obs = assemble_obs(proprio, hist)
        # yaw writeback happens inside policy.act
        action = policy.act(obs)
        if trace is not None:
            _roll, _pitch, yaw = quat_wxyz_to_euler(data.qpos[3:7])
            vis = policy.yaw.detach().cpu().numpy().reshape(-1)
            trace.append(
                {
                    "step": int(step),
                    "x": float(data.qpos[0]),
                    "y": float(data.qpos[1]),
                    "z": float(data.qpos[2]),
                    "yaw": float(yaw),
                    "obs6": float(1.5 * vis[0]),
                    "obs7": float(1.5 * vis[1]),
                    "fr_y": float(data.geom_xpos[foot_ids[0]][1]),
                    "fl_y": float(data.geom_xpos[foot_ids[1]][1]),
                }
            )
        if step == 0:
            first_action = action.tolist()

        if np.any(np.abs(action) >= 4.79):
            saturated += 1

        # action-rate metric on raw policy output (pre-EMA)
        action_rates.append(float(np.linalg.norm(action - prev_action_for_rate)))
        prev_action_snapshot = prev_action_for_rate.copy()
        prev_action_for_rate = action.copy()

        cmd_action = action.copy()
        if action_ema is not None:
            # low-pass: smoothed = (1-ema)*smoothed + ema*new  (ema small → heavier smoothing)
            smoothed_action = (1.0 - action_ema) * smoothed_action + action_ema * cmd_action
            cmd_action = smoothed_action.copy()

        action_queue.append(cmd_action)
        applied_action = action_queue.pop(0)
        last_action = action.copy()  # obs last_action = policy output (not delayed)

        if front_action_scale is not None:
            scale = np.full(12, action_scale, dtype=np.float64)
            scale[0:6] = front_action_scale
            target = DEFAULT_DOF + scale * applied_action
        else:
            target = DEFAULT_DOF + action_scale * applied_action

        kd_vec = np.full(12, kd, dtype=np.float64)
        if front_kd_scale is not None:
            kd_vec[0:6] = front_kd

        # --- physics decimation ---
        # ctrl is actuator order FR,FL,RR,RL — same as policy action order
        for _ in range(decimation):
            qj = data.qpos[QPOS_IDX].copy()
            dq = data.qvel[QVEL_IDX].copy()
            tau = kp * (target - qj) - kd_vec * dq
            data.ctrl[:] = np.clip(tau, -torque_lim, torque_lim)
            mujoco.mj_step(model, data)

        # metrics
        bx = float(data.qpos[0])
        by = float(data.qpos[1])
        bz = float(data.qpos[2])
        roll, pitch, _ = quat_wxyz_to_euler(data.qpos[3:7])
        ang = proprio[0:3] / 0.25  # undo obs scale → rad/s
        max_x = max(max_x, bx)
        max_z = max(max_z, bz)
        rolls.append(roll)
        pitches.append(pitch)
        ys.append(by)
        angvels.append(float(np.linalg.norm(ang)))

        # --- flail metrics ---
        dq = data.qvel[QVEL_IDX].copy()
        joint_vel_norms.append(float(np.linalg.norm(dq)))
        joint_vel_peaks.append(float(np.max(np.abs(dq))))
        # calf joints are indices 2,5,8,11 in FR/FL/RR/RL policy order
        calf_vel_abs.append(float(np.mean(np.abs(dq[[2, 5, 8, 11]]))))
        contacts_now = foot_contacts(model, data, foot_ids)
        front_sw_step: List[float] = []
        rear_sw_step: List[float] = []
        for fi, gid in enumerate(foot_ids):
            if gid < 0:
                continue
            fx, fy, fz = [float(v) for v in data.geom_xpos[gid]]
            th = terrain_height_xy(fx, fy)
            clear = fz - th - float(model.geom_size[gid][0])  # center→bottom ≈ radius
            stance_tot += 1
            if contacts_now[fi] > 0.0:
                stance_hits += 1
            else:
                # swing clearance (can be slightly negative on near-contact)
                swing_clears.append(clear)
                if fi < 2:
                    front_sw_step.append(clear)
                else:
                    rear_sw_step.append(clear)

        on = (
            PASS_X[0] <= bx <= PASS_X[1]
            and PASS_Z[0] <= bz <= PASS_Z[1]
            and abs(roll) < PASS_ROLL
            and abs(pitch) < PASS_PITCH
        )
        if on:
            streak += 1
            on_deck += 1
            best_streak = max(best_streak, streak)
            rolls_on.append(roll)
            pitches_on.append(pitch)
            zs_on.append(bz)
            if first_on_step < 0:
                first_on_step = step
                climbed = True
            if climbed and not hold_broken:
                post_climb_hold_cur += 1
                post_climb_hold = max(post_climb_hold, post_climb_hold_cur)
            if climbed:
                post_climb_on_total += 1
        else:
            streak = 0
            if climbed and not hold_broken and first_on_step >= 0:
                # break continuous hold after climb
                if step > first_on_step:
                    hold_broken = True
                    post_climb_hold_cur = 0
            # fell after climb: was on deck, now low and past lip
            if climbed and bz < 0.45 and bx > PASS_X[0]:
                fell_after_climb = 1

        # --- phase × leg (approach / preclimb / climb / ondeck) ---
        if on:
            ph = "ondeck"
        elif bx < 0.70:
            ph = "approach"
        elif bx < 1.00:
            ph = "preclimb"
        else:
            ph = "climb"
        f_calf = float(np.mean(np.abs(dq[[2, 5]])))
        r_calf = float(np.mean(np.abs(dq[[8, 11]])))
        ph_front_calf[ph].append(f_calf)
        ph_rear_calf[ph].append(r_calf)
        all_front_calf.append(f_calf)
        all_rear_calf.append(r_calf)
        ph_front_jvel[ph].append(float(np.linalg.norm(dq[0:6])))
        ph_rear_jvel[ph].append(float(np.linalg.norm(dq[6:12])))
        ph_pitch[ph].append(abs(pitch))
        ph_front_arate[ph].append(float(np.linalg.norm(action[0:6] - prev_action_snapshot[0:6])))
        ph_front_swing[ph].extend(front_sw_step)
        ph_rear_swing[ph].extend(rear_sw_step)
        all_front_swing.extend(front_sw_step)
        all_rear_swing.extend(rear_sw_step)

        if renderer is not None and step % 2 == 0:
            # 3/4 follow camera
            sc = mujoco.MjvCamera()
            sc.type = mujoco.mjtCamera.mjCAMERA_FREE
            sc.lookat[:] = [bx + 0.9, 0.0, 0.30 + (bz - 0.32) * 0.3]
            sc.distance = 3.6
            sc.azimuth = 135
            sc.elevation = -18
            renderer.update_scene(data, sc)
            frames.append(renderer.render().copy())

    if renderer is not None:
        renderer.close()
        if frames and record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = record_path.with_suffix(".raw.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(tmp), fourcc, 25.0, (1280, 720))
            for fr in frames:
                bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                vw.write(bgr)
            vw.release()
            # ffmpeg re-encode if available
            out = record_path
            rc = os.system(
                f'ffmpeg -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p '
                f'-crf 20 -movflags +faststart "{out}" && rm -f "{tmp}"'
            )
            if rc != 0 and tmp.exists():
                tmp.rename(out)

    # end-on-deck check
    bx = float(data.qpos[0])
    bz = float(data.qpos[2])
    roll, pitch, _ = quat_wxyz_to_euler(data.qpos[3:7])
    end_on = int(
        PASS_X[0] <= bx <= PASS_X[1]
        and PASS_Z[0] <= bz <= PASS_Z[1]
        and abs(roll) < PASS_ROLL
        and abs(pitch) < PASS_PITCH
    )

    passed = 1 if best_streak >= PASS_STREAK else 0
    if hasattr(cam, "close"):
        cam.close()
    return EpisodeResult(
        cmd=cmd_x,
        config=cfg_name,
        trial=trial,
        passed=passed,
        max_x=float(max_x),
        max_z=float(max_z),
        last_x=float(data.qpos[0]),
        last_z=float(data.qpos[2]),
        best_streak=int(best_streak),
        on_deck=int(on_deck),
        saturated=int(saturated),
        first_latent_norm=float(first_latent),
        first_action=first_action,
        depth_updates=depth_updates,
        steps=n_steps,
        roll_rms=_rms(rolls),
        pitch_rms=_rms(pitches),
        roll_rms_on=_rms(rolls_on),
        pitch_rms_on=_rms(pitches_on),
        angvel_rms=_rms(angvels),
        y_rms=_rms(ys),
        max_abs_y=float(np.max(np.abs(ys))) if ys else float("nan"),
        z_std_on=float(np.std(zs_on)) if len(zs_on) > 1 else float("nan"),
        first_on_step=int(first_on_step),
        post_climb_hold=int(post_climb_hold),
        post_climb_on_total=int(post_climb_on_total),
        fell_after_climb=int(fell_after_climb),
        end_on_deck=int(end_on),
        joint_vel_rms=_rms(joint_vel_norms),
        joint_vel_peak=float(np.max(joint_vel_peaks)) if joint_vel_peaks else float("nan"),
        calf_vel_rms=_rms(calf_vel_abs),
        action_rate_rms=_rms(action_rates),
        swing_clear_mean=float(np.mean(swing_clears)) if swing_clears else float("nan"),
        swing_clear_p95=(
            float(np.percentile(swing_clears, 95)) if swing_clears else float("nan")
        ),
        swing_clear_max=float(np.max(swing_clears)) if swing_clears else float("nan"),
        stance_frac=float(stance_hits / stance_tot) if stance_tot else float("nan"),
        approach_front_calf_rms=_rms(ph_front_calf["approach"]),
        approach_rear_calf_rms=_rms(ph_rear_calf["approach"]),
        approach_front_swing_p95=(
            float(np.percentile(ph_front_swing["approach"], 95))
            if ph_front_swing["approach"]
            else float("nan")
        ),
        approach_rear_swing_p95=(
            float(np.percentile(ph_rear_swing["approach"], 95))
            if ph_rear_swing["approach"]
            else float("nan")
        ),
        approach_front_jvel_rms=_rms(ph_front_jvel["approach"]),
        approach_rear_jvel_rms=_rms(ph_rear_jvel["approach"]),
        approach_pitch_rms=_rms(ph_pitch["approach"]),
        preclimb_front_calf_rms=_rms(ph_front_calf["preclimb"]),
        preclimb_rear_calf_rms=_rms(ph_rear_calf["preclimb"]),
        preclimb_front_swing_p95=(
            float(np.percentile(ph_front_swing["preclimb"], 95))
            if ph_front_swing["preclimb"]
            else float("nan")
        ),
        preclimb_rear_swing_p95=(
            float(np.percentile(ph_rear_swing["preclimb"], 95))
            if ph_rear_swing["preclimb"]
            else float("nan")
        ),
        preclimb_front_jvel_rms=_rms(ph_front_jvel["preclimb"]),
        preclimb_rear_jvel_rms=_rms(ph_rear_jvel["preclimb"]),
        preclimb_pitch_rms=_rms(ph_pitch["preclimb"]),
        preclimb_front_action_rate_rms=_rms(ph_front_arate["preclimb"]),
        climb_front_calf_rms=_rms(ph_front_calf["climb"]),
        climb_rear_calf_rms=_rms(ph_rear_calf["climb"]),
        climb_front_swing_p95=(
            float(np.percentile(ph_front_swing["climb"], 95))
            if ph_front_swing["climb"]
            else float("nan")
        ),
        ondeck_front_calf_rms=_rms(ph_front_calf["ondeck"]),
        ondeck_rear_calf_rms=_rms(ph_rear_calf["ondeck"]),
        ondeck_front_swing_p95=(
            float(np.percentile(ph_front_swing["ondeck"], 95))
            if ph_front_swing["ondeck"]
            else float("nan")
        ),
        ondeck_pitch_rms=_rms(ph_pitch["ondeck"]),
        front_calf_rms=_rms(all_front_calf),
        rear_calf_rms=_rms(all_rear_calf),
        front_swing_p95=(
            float(np.percentile(all_front_swing, 95)) if all_front_swing else float("nan")
        ),
        rear_swing_p95=(
            float(np.percentile(all_rear_swing, 95)) if all_rear_swing else float("nan")
        ),
    )


def load_model(scene: Path, cfg: ContactConfig) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = DT
    apply_contact_config(model, cfg)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--vision", type=Path, default=DEFAULT_VISION)
    ap.add_argument("--jit", type=Path, default=DEFAULT_JIT)
    ap.add_argument("--config", type=str, default="baseline")
    ap.add_argument("--cmd", type=float, default=1.2)
    ap.add_argument("--cmds", type=str, default="0.8,1.0,1.1,1.2,1.3")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--configs", type=str, default="")
    args = ap.parse_args()

    if args.config not in CONFIGS and not args.configs:
        raise SystemExit(f"unknown config {args.config}; choose from {list(CONFIGS)}")

    cfg_names = (
        [c.strip() for c in args.configs.split(",") if c.strip()]
        if args.configs
        else [args.config]
    )
    cmds = [float(x) for x in args.cmds.split(",")] if args.grid else [args.cmd]

    args.out.mkdir(parents=True, exist_ok=True)
    vid_dir = HERE / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    policy = EPStudentPolicy(args.vision, args.jit, device=args.device)
    all_results: List[dict] = []

    for cname in cfg_names:
        cfg = CONFIGS[cname]
        print(f"\n=== CONFIG {cname}: {cfg.note} ===", flush=True)
        model = load_model(args.scene, cfg)
        for cmd in cmds:
            passes = 0
            for t in range(args.trials):
                rec = None
                if args.record and t == 0:
                    tag = f"{int(round(cmd * 10)):02d}"
                    rec = vid_dir / f"mujoco-gait-{cname}-cmd{tag}.mp4"
                t0 = time.time()
                # re-new model data path: reload model each trial for determinism
                model = load_model(args.scene, cfg)
                policy = EPStudentPolicy(args.vision, args.jit, device=args.device)
                res = run_episode(
                    model,
                    policy,
                    cmd,
                    cname,
                    t,
                    n_steps=args.steps,
                    record_path=rec,
                    cfg=cfg,
                )
                passes += res.passed
                d = asdict(res)
                d["wall_s"] = time.time() - t0
                if rec is not None:
                    d["video"] = str(rec)
                all_results.append(d)
                print(
                    f"[EP_RESULT] cfg={cname} cmd={cmd} trial={t} pass={res.passed} "
                    f"max_x={res.max_x:.5f} max_z={res.max_z:.5f} streak={res.best_streak} "
                    f"on_deck={res.on_deck} hold={res.post_climb_hold} "
                    f"roll_rms={res.roll_rms:.4f} pitch_rms={res.pitch_rms:.4f} "
                    f"roll_on={res.roll_rms_on:.4f} pitch_on={res.pitch_rms_on:.4f} "
                    f"angvel={res.angvel_rms:.4f} y_rms={res.y_rms:.4f} "
                    f"jvel={res.joint_vel_rms:.3f} jpeak={res.joint_vel_peak:.2f} "
                    f"calf={res.calf_vel_rms:.3f} arate={res.action_rate_rms:.3f} "
                    f"sw_m={res.swing_clear_mean:.3f} sw_p95={res.swing_clear_p95:.3f} "
                    f"sw_max={res.swing_clear_max:.3f} stance={res.stance_frac:.2f} "
                    f"fell_ac={res.fell_after_climb} sat={res.saturated} "
                    f"last=({res.last_x:.3f},{res.last_z:.3f}) {time.time()-t0:.1f}s",
                    flush=True,
                )
            print(f"  >> cmd={cmd} pass={passes}/{args.trials}", flush=True)

    tag = "_".join(cfg_names)
    if len(tag) > 80:
        tag = f"n{len(cfg_names)}_{cfg_names[0]}_{cfg_names[-1]}"
    out_json = args.out / f"results_{tag}_{int(time.time())}.json"
    summary = {
        "results": all_results,
        "configs": {k: asdict(CONFIGS[k]) for k in cfg_names},
        "scene": str(args.scene),
        "pass_criterion": {
            "x": PASS_X,
            "z": PASS_Z,
            "roll": PASS_ROLL,
            "pitch": PASS_PITCH,
            "streak": PASS_STREAK,
        },
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
