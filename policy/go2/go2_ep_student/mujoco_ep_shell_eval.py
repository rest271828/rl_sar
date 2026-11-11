#!/usr/bin/env python3
"""Local MuJoCo plane-depth closed-loop shell for go2_ep_student.

Replaces the A6000-only launcher target
`/home/yihan/agent_ws/experiments/ep_on_shen_shell/mujoco_ep_shen_shell_eval.py`
which is not present on this laptop (SSH to cs2.sg:3026 times out on banner).

Contract matches A6000 notes / C++ plane-depth patch:
  student JIT + vision encoder, depth every update_interval=5,
  scene_step40_stable_v6 (stable_isaac_v6 damping), action CSV dump,
  front-leg |a| clip stats vs ±4.8.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ep_depth_encoder import RecurrentDepthBackbone  # noqa: E402

import mujoco  # noqa: E402

N_PROPRIO = 53
N_OBS = 753
UPDATE_INTERVAL = 5
ACTION_CLIP = 4.8
YAW_SCALE = 1.5
ACTION_SCALE = 0.25
KP = 40.0
KD = 1.0
DECIMATION = 10  # XML timestep 0.002 → 50 Hz policy

DEPTH_H, DEPTH_W = 60, 106
CROP_H, CROP_W = 58, 98
OUT_H, OUT_W = 58, 87
FAR = 2.0
HFOV_DEG = 87.0
PITCH_DEG = 22.9
CAM_BODY = np.array([0.24, -0.0175, 0.12], dtype=np.float64)

DEFAULT_DOF = np.array(
    [-0.1, 0.7, -1.5, 0.1, 0.7, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5],
    dtype=np.float32,
)
# Front thigh/calf indices in FR,FL,RR,RL order
FRONT_LEG_IDX = (1, 2, 4, 5)
FRONT_LEG_NAMES = ("FR_thigh", "FR_calf", "FL_thigh", "FL_calf")


def quat_to_euler_wxyz(q: np.ndarray) -> Tuple[float, float, float]:
    w, x, y, z = [float(v) for v in q]
    # roll (x), pitch (y), yaw (z)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def build_encoder(vision_path: Path, device: torch.device) -> RecurrentDepthBackbone:
    enc = RecurrentDepthBackbone(n_proprio=N_PROPRIO).to(device)
    blob = torch.load(str(vision_path), map_location=device, weights_only=False)
    if isinstance(blob, dict) and "depth_encoder_state_dict" in blob:
        sd = blob["depth_encoder_state_dict"]
    else:
        sd = blob
    missing, unexpected = enc.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"vision load mismatch missing={missing} unexpected={unexpected}")
    enc.eval()
    return enc


class PlaneDepth:
    """MuJoCo mj_multiRay plane-depth → EP 58x87 (meters/far - 0.5)."""

    def __init__(self) -> None:
        hfov = math.radians(HFOV_DEG)
        fx = 0.5 * DEPTH_W / math.tan(0.5 * hfov)
        fy = fx
        cx = 0.5 * (DEPTH_W - 1)
        cy = 0.5 * (DEPTH_H - 1)
        dirs = np.zeros((DEPTH_H * DEPTH_W, 3), dtype=np.float64)
        plane_scale = np.zeros(DEPTH_H * DEPTH_W, dtype=np.float64)
        for v in range(DEPTH_H):
            for u in range(DEPTH_W):
                i = v * DEPTH_W + u
                x = (u - cx) / fx
                y = -(v - cy) / fy
                z = -1.0
                nrm = math.sqrt(x * x + y * y + z * z)
                x, y, z = x / nrm, y / nrm, z / nrm
                dirs[i] = (x, y, z)
                plane_scale[i] = -z
        self.dirs_cam = dirs
        self.plane_scale = plane_scale

    def render(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if bid < 0:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if bid < 0:
            raise RuntimeError("base_link/base body not found")
        xpos = data.xpos[bid].copy()
        xmat = data.xmat[bid].reshape(3, 3).copy()

        pitch = math.radians(PITCH_DEG)
        forward = np.array([math.cos(pitch), 0.0, -math.sin(pitch)], dtype=np.float64)
        right = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        z_cam = -forward
        up = np.cross(z_cam, right)
        up /= np.linalg.norm(up)
        r_cb = np.column_stack([right, up, z_cam])  # body←cam columns
        r_cw = xmat @ r_cb

        cam_body_w = xmat @ CAM_BODY
        pnt = xpos + cam_body_w

        nray = DEPTH_H * DEPTH_W
        dirs_w = (self.dirs_cam @ r_cw.T).reshape(-1)  # row vectors → world
        geomgroup = np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8)
        geomid = np.full(nray, -1, dtype=np.int32)
        dist = np.full(nray, -1.0, dtype=np.float64)
        mujoco.mj_multiRay(
            model,
            data,
            pnt,
            dirs_w,
            geomgroup,
            1,  # flg_static
            bid,  # bodyexclude
            geomid,
            dist,
            nray,
            FAR * 2.0,
        )

        cropped = np.empty((CROP_H, CROP_W), dtype=np.float32)
        for v in range(CROP_H):
            for u in range(CROP_W):
                src = v * DEPTH_W + (u + 4)
                depth = FAR
                if dist[src] >= 0.0:
                    depth = min(dist[src] * self.plane_scale[src], FAR)
                if depth < 0.0:
                    depth = 0.0
                cropped[v, u] = depth

        # bicubic resize to 58x87 via torch for fidelity/speed
        t = torch.from_numpy(cropped)[None, None]  # 1,1,H,W
        resized = torch.nn.functional.interpolate(
            t, size=(OUT_H, OUT_W), mode="bicubic", align_corners=False
        )
        out = resized.squeeze().numpy().astype(np.float32)
        out = out / float(FAR) - 0.5
        return out


def resolve_scene(config: str) -> Path:
    scenes = _REPO / "scenes" / "go2_ep_step40"
    mapping = {
        "stable_isaac_v6": scenes / "scene_step40_stable_v6.xml",
        "stable_v6": scenes / "scene_step40_stable_v6.xml",
        "stable_v4": scenes / "scene_step40_stable_v4.xml",
        "stable_v2": scenes / "scene_step40_stable_v2.xml",
        "fri15": scenes / "scene_step40_fri15.xml",
        "step40": scenes / "scene_step40.xml",
    }
    path = mapping.get(config)
    if path is None:
        raise SystemExit(f"unknown --config {config}; choose {sorted(mapping)}")
    if not path.is_file():
        raise SystemExit(f"scene missing: {path}")
    return path


def sensor_adr(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise RuntimeError(f"sensor {name} missing")
    return int(model.sensor_adr[sid])


class Go2Shell:
    def __init__(
        self,
        scene: Path,
        base_jit: Path,
        vision: Path,
        cmd_x: float,
        device: str,
        dump_path: Optional[Path],
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.cmd_x = float(cmd_x)
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.encoder = build_encoder(vision, self.device)
        self.policy = torch.jit.load(str(base_jit), map_location=self.device)
        self.policy.eval()
        self.plane = PlaneDepth()
        self.dump_path = dump_path
        self.hist: Deque[np.ndarray] = deque(maxlen=10)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.latent = torch.zeros(1, 32, device=self.device)
        self.yaw = torch.zeros(1, 2, device=self.device)
        self.policy_step = 0

        # sensor addresses (FR…RL order matches actuators)
        self.q_adr = [sensor_adr(self.model, n) for n in [
            "FR_hip_pos", "FR_thigh_pos", "FR_calf_pos",
            "FL_hip_pos", "FL_thigh_pos", "FL_calf_pos",
            "RR_hip_pos", "RR_thigh_pos", "RR_calf_pos",
            "RL_hip_pos", "RL_thigh_pos", "RL_calf_pos",
        ]]
        self.dq_adr = [sensor_adr(self.model, n) for n in [
            "FR_hip_vel", "FR_thigh_vel", "FR_calf_vel",
            "FL_hip_vel", "FL_thigh_vel", "FL_calf_vel",
            "RR_hip_vel", "RR_thigh_vel", "RR_calf_vel",
            "RL_hip_vel", "RL_thigh_vel", "RL_calf_vel",
        ]]
        self.quat_adr = sensor_adr(self.model, "imu_quat")
        self.gyro_adr = sensor_adr(self.model, "imu_gyro")
        self.base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        # MJCF body-tree joint order is FL,FR,RL,RR — not actuator/policy FR,FL,RR,RL.
        joint_names_policy = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        self.joint_qposadr = []
        for name in joint_names_policy:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint {name} missing")
            self.joint_qposadr.append(int(self.model.jnt_qposadr[jid]))

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # Prefer home keyframe, then overwrite freejoint + policy-default stand pose.
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[0] = float(getattr(self, "spawn_x", 0.2))
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.32
        self.data.qpos[3:7] = np.array([1, 0, 0, 0], dtype=np.float64)
        for adr, q0 in zip(self.joint_qposadr, DEFAULT_DOF):
            self.data.qpos[adr] = float(q0)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.hist.clear()
        self.last_action[:] = 0
        self.encoder.reset_hidden()
        self.latent.zero_()
        self.yaw.zero_()
        self.policy_step = 0
        # Settle under stand PD (zero action) so feet find contact before RL.
        for _ in range(250):
            self.apply_pd(DEFAULT_DOF.copy())
            mujoco.mj_step(self.model, self.data)

    def read_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = np.array([self.data.sensordata[a] for a in self.q_adr], dtype=np.float32)
        dq = np.array([self.data.sensordata[a] for a in self.dq_adr], dtype=np.float32)
        quat = np.array(self.data.sensordata[self.quat_adr:self.quat_adr + 4], dtype=np.float32)
        gyro = np.array(self.data.sensordata[self.gyro_adr:self.gyro_adr + 3], dtype=np.float32)
        return q, dq, quat, gyro

    def build_proprio(self, q, dq, quat, gyro) -> np.ndarray:
        proprio = np.zeros(N_PROPRIO, dtype=np.float32)
        proprio[0:3] = gyro * 0.25
        roll, pitch, _ = quat_to_euler_wxyz(quat)
        proprio[3] = roll
        proprio[4] = pitch
        # 5,6,7 yaw slots; 8,9 cmd masked; 10 = vx
        proprio[10] = self.cmd_x
        proprio[11] = 1.0  # parkour / non-flat (step40)
        proprio[12] = 0.0
        proprio[13:25] = (q - DEFAULT_DOF) * 1.0
        proprio[25:37] = dq * 0.05
        proprio[37:49] = self.last_action
        proprio[49:53] = -0.5
        return proprio

    def build_obs(self, proprio: np.ndarray) -> torch.Tensor:
        hist_frame = proprio.copy()
        hist_frame[6:8] = 0.0
        if len(self.hist) == 0:
            for _ in range(10):
                self.hist.append(hist_frame.copy())
        else:
            self.hist.append(hist_frame.copy())
        obs = np.zeros(N_OBS, dtype=np.float32)
        obs[:N_PROPRIO] = proprio
        hist_off = N_PROPRIO + 132 + 9 + 29
        hist = np.concatenate(list(self.hist), axis=0)
        obs[hist_off:hist_off + 10 * N_PROPRIO] = hist
        np.clip(obs, -100.0, 100.0, out=obs)
        return torch.from_numpy(obs).unsqueeze(0).to(self.device)

    def apply_pd(self, q_des: np.ndarray) -> None:
        q, dq, _, _ = self.read_state()
        tau = KP * (q_des - q) - KD * dq
        # Hip/thigh 23.5; calf/knee XML allows ~45 (matches EP knee motor).
        lim = np.array([23.5, 23.5, 45.0] * 4, dtype=np.float32)
        tau = np.clip(tau, -lim, lim)
        self.data.ctrl[:] = tau

    @torch.no_grad()
    def policy_step_once(self) -> Tuple[np.ndarray, float]:
        q, dq, quat, gyro = self.read_state()
        proprio = self.build_proprio(q, dq, quat, gyro)
        obs = self.build_obs(proprio)

        if getattr(self, "use_depth", True) and self.policy_step % UPDATE_INTERVAL == 0:
            depth = self.plane.render(self.model, self.data)
            depth_t = torch.from_numpy(depth).unsqueeze(0).to(self.device)
            prop_t = obs[:, :N_PROPRIO].clone()
            prop_t[:, 6:8] = 0
            depth_out = self.encoder(depth_t, prop_t)
            self.latent = depth_out[:, :-2]
            self.yaw = depth_out[:, -2:]

        obs[:, 6:8] = YAW_SCALE * self.yaw
        action = self.policy(obs, self.latent)
        action = action.clamp(-ACTION_CLIP, ACTION_CLIP)
        a = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        self.last_action = a
        self.policy_step += 1
        q_des = DEFAULT_DOF + ACTION_SCALE * a
        return a, float(np.abs(a).max())

    def base_pose(self) -> Tuple[float, float, float, float, float]:
        xpos = self.data.xpos[self.base_bid]
        xmat = self.data.xmat[self.base_bid].reshape(3, 3)
        # approximate pitch/roll from rotation matrix
        pitch = math.asin(max(-1.0, min(1.0, -xmat[2, 0])))
        roll = math.atan2(xmat[2, 1], xmat[2, 2])
        return float(xpos[0]), float(xpos[1]), float(xpos[2]), float(roll), float(pitch)

    def run(self, seconds: float) -> dict:
        self.reset()
        dt = float(self.model.opt.timestep)
        n_steps = int(seconds / dt)
        rows: List[list] = []
        z_max = 0.0
        climbed = False
        front_abs: List[float] = []

        for i in range(n_steps):
            if i % DECIMATION == 0:
                a, amax = self.policy_step_once()
                x, y, z, roll, pitch = self.base_pose()
                z_max = max(z_max, z)
                if z >= 0.55 and x >= 1.2:
                    climbed = True
                for j in FRONT_LEG_IDX:
                    front_abs.append(abs(float(a[j])))
                if self.dump_path is not None:
                    rows.append(
                        [i * dt, self.policy_step - 1, x, y, z, roll, pitch, amax, *a.tolist()]
                    )
            # hold last q_des between policy ticks
            q_des = DEFAULT_DOF + ACTION_SCALE * self.last_action
            self.apply_pd(q_des)
            mujoco.mj_step(self.model, self.data)

        if self.dump_path is not None:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
            with self.dump_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    ["t", "policy_step", "x", "y", "z", "roll", "pitch", "amax"]
                    + [f"a{i}" for i in range(12)]
                )
                w.writerows(rows)

        front = np.array(front_abs, dtype=np.float64) if front_abs else np.zeros(1)
        x, y, z, roll, pitch = self.base_pose()
        return {
            "seconds": seconds,
            "policy_steps": self.policy_step,
            "z_max": z_max,
            "final_xyz": (x, y, z),
            "final_rpy": (roll, pitch),
            "climbed": bool(climbed and z_max >= 0.55 and x >= 1.2),
            "front_abs_mean": float(front.mean()),
            "front_abs_p95": float(np.percentile(front, 95)),
            "front_abs_max": float(front.max()),
            "front_clip_ge_4_75": float(np.mean(front >= 4.75)),
            "front_clip_ge_4_0": float(np.mean(front >= 4.0)),
            "dump": str(self.dump_path) if self.dump_path else "",
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Go2 EP student MuJoCo plane-depth shell (local)")
    ap.add_argument("--config", default="stable_isaac_v6",
                    help="scene alias; default stable_isaac_v6 → scene_step40_stable_v6.xml")
    ap.add_argument("--cmd-x", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--base-jit", default=str(_HERE / "base_jit.pt"))
    ap.add_argument("--vision-weight", default=str(_HERE / "vision_weight.pt"))
    ap.add_argument("--dump-actions", default="",
                    help="CSV path for per-policy-step action dump")
    ap.add_argument("--scene", default="", help="override MJCF path")
    ap.add_argument("--spawn-x", type=float, default=0.2, help="freejoint x before step (front at x=1.0)")
    ap.add_argument("--no-depth", action="store_true", help="keep latent=0 (ablation)")
    ap.add_argument("--ok-exit", action="store_true", help="always exit 0 after printing metrics")
    args = ap.parse_args(argv)

    scene = Path(args.scene) if args.scene else resolve_scene(args.config)
    dump = Path(args.dump_actions) if args.dump_actions else None
    print(f"scene={scene}")
    print(f"base_jit={args.base_jit}")
    print(f"vision={args.vision_weight}")
    print(f"cmd_x={args.cmd_x} seconds={args.seconds} device={args.device} spawn_x={args.spawn_x} depth={not args.no_depth}")

    shell = Go2Shell(
        scene=scene,
        base_jit=Path(args.base_jit),
        vision=Path(args.vision_weight),
        cmd_x=args.cmd_x,
        device=args.device,
        dump_path=dump,
    )
    shell.spawn_x = args.spawn_x
    shell.use_depth = not args.no_depth
    stats = shell.run(args.seconds)
    print(
        "RESULT climbed={climbed} z_max={z_max:.3f} final_xyz=({fx:.3f},{fy:.3f},{fz:.3f}) "
        "front_|a|_mean={front_abs_mean:.3f} p95={front_abs_p95:.3f} max={front_abs_max:.3f} "
        "clip>=4.75={front_clip_ge_4_75:.3%} clip>=4.0={front_clip_ge_4_0:.3%} "
        "policy_steps={policy_steps} dump={dump}".format(
            climbed=stats["climbed"],
            z_max=stats["z_max"],
            fx=stats["final_xyz"][0],
            fy=stats["final_xyz"][1],
            fz=stats["final_xyz"][2],
            front_abs_mean=stats["front_abs_mean"],
            front_abs_p95=stats["front_abs_p95"],
            front_abs_max=stats["front_abs_max"],
            front_clip_ge_4_75=stats["front_clip_ge_4_75"],
            front_clip_ge_4_0=stats["front_clip_ge_4_0"],
            policy_steps=stats["policy_steps"],
            dump=stats["dump"] or "-",
        )
    )
    if args.ok_exit:
        return 0
    return 0 if stats["climbed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
