# Simulation entrypoints (Go2 / go2_ep_student)

## rl_sar stock sim (himloco / robot_lab)

These run the **existing** locomotion policies (45-dim obs, single TorchScript). They do **not** load EP Student.

### Gazebo + `rl_sim` (ROS required)

```bash
# Terminal 1
roslaunch rl_sar gazebo.launch rname:=go2
# or ROS2:
#   ros2 launch rl_sar gazebo.launch.py rname:=go2

# Terminal 2
rosrun rl_sar rl_sim
# or: ros2 run rl_sar rl_sim
```

FSM default for Go2 locomotion: `config_name = "himloco"` (`fsm_go2.hpp`). Policy files: `policy/go2/himloco/`.

### MuJoCo (CMake `-mj` build)

```bash
./build.sh -mj
./cmake_build/bin/rl_sim_mujoco go2 scene
```

### Why EP Student is not plugged into Gazebo yet

- `RL_Sim::Forward()` → `model->forward({obs})` **single** tensor.
- EP needs `(obs[753], depth_latent[32])` + live depth GRU (`vision_weight.pt`).
- Gazebo Go2 stack has **no** depth camera → encoder path.
- Host `cs2.sg` currently: no `/opt/ros`, no mujoco lib, docker socket denied → stock rl_sar sim not runnable here without setup.

Selecting `go2_ep_student` via `InitRL("go2/go2_ep_student")` would load `base_jit.pt` then **crash/mis-infer** on single-input Forward. Do not switch FSM to that name until Forward is extended.

## go2_ep_student sim path (this repo slot)

### A) Offline dummy (no physics)

```bash
source /home/yihan/extreme-parkour/activate.sh
cd /home/yihan/extreme-parkour/rest271828-rl_sar
python policy/go2/go2_ep_student/infer_offline.py
```

### B) Isaac Gym closed-loop bridge (preferred smoke on cs2)

Uses Extreme Parkour Go2 sim + this sidecar’s JIT/vision (same weights as deploy).

```bash
source /home/yihan/extreme-parkour/activate.sh
cd /home/yihan/extreme-parkour/rest271828-rl_sar
python policy/go2/go2_ep_student/sim_isaac_bridge.py --steps 200 --num-envs 4
# longer / parkour + evaluate-like summary (MXD_proxy = waypoints/7):
python policy/go2/go2_ep_student/sim_isaac_bridge.py --backend jit --steps 1500 --num-envs 192 --terrain parkour
# evaluate-parity (depth_actor from ckpt):
python policy/go2/go2_ep_student/sim_isaac_bridge.py --backend ckpt --steps 1500 --num-envs 192 --terrain parkour
```

### Next sim gap (not this change)

1. Extend `rl_sim` / `rl_real_go2` `Forward()` for dual-input + depth topic, **or** keep Python controller alongside Gazebo.
2. Add depth camera to Gazebo Go2 and match EP preprocess (87×58, buffer `-2`).
3. Optional FSM key to select `himloco` vs future EP state without overwriting himloco files.
