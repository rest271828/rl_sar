# Sim entrypoints

| Path | Command | Depth | Notes |
|---|---|---|---|
| MuJoCo rl_sar shell | `RL_SAR_GO2_CONFIG=go2_ep_student RL_SAR_AUTO_RL=1 bash scripts/run_go2_mujoco_xvfb.sh` | zero latent | C++ dual-input Forward |
| MuJoCo smoke | `bash scripts/smoke_go2_ep_mujoco.sh` | zero latent | auto FSM |
| Isaac bridge JIT | `sim_isaac_bridge.py --backend jit ...` | interval-5 encoder | deploy graph |
| Isaac bridge ckpt | `sim_isaac_bridge.py --backend ckpt ...` | same as evaluate | MXD≈0.88 |
