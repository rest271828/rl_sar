# go2_ep_student — Extreme Parkour Student plugin (sidecar)

New policy slot under `policy/go2/`. **Does not overwrite** `himloco` or `robot_lab`.

## What this is

- **Weights:** EP Student `922-10-GO2-STUDENT-D0` / `model_59500`
  - `base_jit.pt` — TorchScript `(obs[753], depth_latent[32]) → action[12]`
  - `vision_weight.pt` — `depth_encoder_state_dict` (GRU, not traced)
- **Offline proof:** `infer_offline.py` — dummy depth+obs, no robot.
- **Sim proof:** `sim_isaac_bridge.py` — Isaac Gym Go2 closed-loop with the sidecar (see `SIM.md`).
- **Not yet:** Gazebo/MuJoCo FSM hookup, ROS topics, real robot.

## Quick commands

```bash
source /home/yihan/extreme-parkour/activate.sh
cd /home/yihan/extreme-parkour/rest271828-rl_sar

# dummy dry-run
python policy/go2/go2_ep_student/infer_offline.py

# sim closed-loop (Isaac Gym; rl_sar Gazebo unavailable on bare cs2)
python policy/go2/go2_ep_student/sim_isaac_bridge.py --steps 200 --num-envs 4
```

## Stock rl_sar Go2 sim (other policies)

```text
roslaunch rl_sar gazebo.launch rname:=go2   # then rosrun rl_sar rl_sim → himloco
./cmake_build/bin/rl_sim_mujoco go2 scene
```

Details and gaps: **SIM.md**. Contract: project doc `ep-student-vs-rl-sar-contract.md`.
