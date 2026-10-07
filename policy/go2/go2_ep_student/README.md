# go2_ep_student

Extreme Parkour Go2 Student (`922-10` / `model_59500`) policy slot for `rest271828/rl_sar`.

Does **not** replace `himloco` / `robot_lab`.

## Artifacts

| File | Role |
|---|---|
| `base_jit.pt` | TorchScript actor `(obs[753], depth_latent[32]) -> action[12]` (symlink to traced) |
| `vision_weight.pt` | RecurrentDepthBackbone weights (Python / Isaac only) |
| `config.yaml` | rl_sar `InitRL` slot (buffer sizing + scales) |

## MuJoCo (`rl_sim_mujoco`) — shell integration

cs2 user-space build (no ROS/sudo):

```bash
# rebuild after C++ changes
export PATH="$HOME/.local/tools/cmake-3.28.3-linux-x86_64/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$("$HOME/.local/bin/micromamba" shell hook -s bash)"
micromamba activate rlsar
cmake --build cmake_build --target rl_sim_mujoco -j$(nproc)

# himloco baseline
RL_SAR_GO2_CONFIG=himloco bash scripts/run_go2_mujoco_xvfb.sh

# EP Student (depth latent = 0 in stock MuJoCo go2 — no camera)
RL_SAR_GO2_CONFIG=go2_ep_student RL_SAR_AUTO_RL=1 bash scripts/run_go2_mujoco_xvfb.sh

# automated smoke
bash scripts/smoke_go2_ep_mujoco.sh
```

Env:

- `RL_SAR_GO2_CONFIG=himloco|robot_lab|go2_ep_student` (default `himloco`)
- `RL_SAR_AUTO_RL=1` — auto Passive→GetUp→RLLocomotion (no keyboard)

C++ changes: multi-input `TorchModel::forward`, EP branch in `RL_Sim::Forward`, FSM `RL_SAR_GO2_CONFIG` / `RL_SAR_AUTO_RL`.

## Isaac Gym — vision MXD

```bash
source /home/yihan/extreme-parkour/activate.sh
cd /home/yihan/extreme-parkour/rest271828-rl_sar
python policy/go2/go2_ep_student/sim_isaac_bridge.py --backend jit --terrain parkour --steps 1500 --num-envs 192
# or: bash scripts/run_go2_ep_student_isaac.sh
```

Parkour MXD (waypoints/7): ckpt≈0.88 (matches evaluate), jit≈0.53 (estimator vs GT priv).

## Limits

- MuJoCo path has **no depth camera** → zero latent; not a vision MXD score.
- Gazebo/`rl_sim` still needs ROS (not on cs2).
- Joystick missing under Xvfb is expected (`/dev/input/js0`).
