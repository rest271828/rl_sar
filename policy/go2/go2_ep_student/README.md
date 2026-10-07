# go2_ep_student

Extreme Parkour Go2 Student (`922-10` / `model_59500`) policy slot for `rest271828/rl_sar`.

Does **not** replace `himloco` / `robot_lab`.

## Artifacts

| File | Role |
|---|---|
| `base_jit.pt` | TorchScript `(obs[753], depth_latent[32]) -> action[12]` (estimator inside) |
| `vision_weight.pt` | RecurrentDepthBackbone weights (Python / Isaac) |
| `config.yaml` | rl_sar `InitRL` + EP scales / `joint_mapping` |

See `WEIGHTS.md` for symlink targets.

## Depth cadence (`update_interval=5`)

**错误节拍** = 把 depth GRU 当成 50 Hz 每步都推，而环境只每 5 个控制步才给新深度帧。正确做法：仅当 `extras["depth"] is not None` 时前进 encoder/GRU，其余步复用 latent/yaw。

Enforced in: `sim_isaac_bridge.py`, `infer_offline.py`. MuJoCo has no camera → latent stays 0 (no GRU).

## Isaac Gym MXD (parkour, 1500×192)

| Backend | MXD (waypoints/7) | Notes |
|---|---:|---|
| `jit` (deploy) | **0.90** | vision_weight + base_jit; estimator; no GT priv |
| `ckpt_est` | **0.90** | depth_actor + estimator (fair twin of jit) |
| `ckpt` | **0.88** | evaluate.py oracle with GT priv_explicit |

```bash
source /home/yihan/extreme-parkour/activate.sh
cd /home/yihan/extreme-parkour/rest271828-rl_sar
python policy/go2/go2_ep_student/sim_isaac_bridge.py --backend jit --terrain parkour --steps 1500 --num-envs 192
```

Critical deploy detail: depth backbone output activation must be **ELU** (training default), not Tanh.

## MuJoCo (`rl_sim_mujoco`)

```bash
RL_SAR_GO2_CONFIG=go2_ep_student RL_SAR_AUTO_RL=1 RL_SAR_CMD_X=0 \
  bash scripts/run_go2_mujoco_xvfb.sh
# or
bash scripts/smoke_go2_ep_mujoco.sh
```

- Dual-input Forward + 753-d EP obs (proprio history, `joint_mapping` FL-first, parkour_flat one-hot)
- `RL_SAR_AUTO_RL=1` locks cmds (blocks Xvfb phantom keys)
- **No depth camera** → zero latent; parkour student is unstable here (not a vision MXD score)

## Limits

- MuJoCo cannot score vision parkour without a depth camera / encoder path
- Gazebo/`rl_sim` still needs ROS
