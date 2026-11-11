# Sim entrypoints

| Path | Command | Depth | Notes |
|---|---|---|---|
| MuJoCo Python shell | `python policy/go2/go2_ep_student/mujoco_ep_shell_eval.py --config stable_isaac_v6 --dump-actions ...` | interval-5 plane rays | **Preferred** local climb + action dump |
| MuJoCo launcher | `python policy/go2/go2_ep_student/sim_mujoco_shell_eval.py ...` | same | Uses A6000 script if present, else local shell |
| C++ `rl_sim_mujoco` | `RL_SAR_GO2_CONFIG=go2_ep_student RL_SAR_AUTO_RL=1 RL_SAR_CMD_X=1.0 RL_SAR_EP_ACTION_DUMP=... ./cmake_build/bin/rl_sim_mujoco go2 scene_step40_stable_v6` | interval-5 plane rays | Needs DISPLAY; dump via env |
| Isaac `jit` | `sim_isaac_bridge.py --backend jit ...` | interval-5 ELU encoder | needs extreme-parkour activate |
| Isaac `ckpt_est` | `--backend ckpt_est` | same | |
| Isaac `ckpt` | `--backend ckpt` | same + GT priv | |

**Depth cadence：** GRU only advances when `update_interval=5` has a new depth frame (not every control step).

**Scene：** `scenes/go2_ep_step40/scene_step40_stable_v6.xml` (`stable_isaac_v6`). Zoo copy: `src/rl_sar_zoo/go2_description/mjcf/scene_step40_stable_v6.xml`.
