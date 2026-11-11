# Sim entrypoints

| Path | Command | Depth | MXD |
|---|---|---|---|
| Isaac `jit` | `sim_isaac_bridge.py --backend jit ...` | interval-5 ELU encoder | **0.90** |
| Isaac `ckpt_est` | `--backend ckpt_est` | same | **0.90** |
| Isaac `ckpt` | `--backend ckpt` | same + GT priv | **0.88** |
| MuJoCo shell | `RL_SAR_GO2_CONFIG=go2_ep_student RL_SAR_AUTO_RL=1 bash scripts/run_go2_mujoco_xvfb.sh` | latent=0 | n/a |
| MuJoCo smoke | `bash scripts/smoke_go2_ep_mujoco.sh` | latent=0 | n/a |
| Recorded `.mm` offline AB | `python policy/go2/go2_ep_student/mm_offline_replay.py --session logs/<sess>_go2_test --modes fill,hold,refuse` | real dump + valid% gate (default 0.2) | n/a |
| C++ depth smoke + gate | `RL_SAR_EP_DEPTH_BIN=... RL_SAR_EP_DEPTH_GATE=refuse ./cmake_build/bin/ep_depth_smoke` | single `.mm`; hold needs `RL_SAR_EP_DEPTH_HOLD_BIN` | n/a |

**错误节拍：** GRU 每控制步前进（错）vs 仅 `update_interval=5` 有新深度时前进（对）。
