# go2_ep_student — Extreme Parkour Student plugin (sidecar)

New policy slot under `policy/go2/`. **Does not overwrite** `himloco` or `robot_lab`.

## What this is

- **Weights:** EP Student `922-10-GO2-STUDENT-D0` / `model_59500`
  - `base_jit.pt` — TorchScript `(obs[753], depth_latent[32]) → action[12]`
  - `vision_weight.pt` — `depth_encoder_state_dict` (GRU, not traced)
- **Offline proof:** `infer_offline.py` loads both and runs dummy depth+obs on CPU/GPU.
- **Not yet:** ROS topics, FSM state, or torque to the robot.

## Dry-run (no robot)

```bash
cd /home/yihan/extreme-parkour/rest271828-rl_sar
source /home/yihan/extreme-parkour/activate.sh
python policy/go2/go2_ep_student/infer_offline.py
```

Optional: copy/symlink traced files into this directory as `base_jit.pt` and `vision_weight.pt`, or pass `--base-jit` / `--vision-weight`.

## Why Python sidecar first

Stock `InferenceRuntime::TorchModel::forward` feeds a **single** obs tensor. EP needs **two** tensors plus a live GRU. Full C++/FSM hookup requires a custom `Forward()` (see upstream README: customize `rl_real_<ROBOT>.cpp`) and a 753-dim observation builder incompatible with himloco’s 45-dim terms.

## Next steps (ROS / FSM)

1. Add `RLFSMStateEPStudent` (or make locomotion `config_name` selectable) — never replace himloco files.
2. Extend Go2 `Forward()` to: preprocess depth → encoder → write yaw into obs[6:8] → `jit(obs, latent)`.
3. Wire RealSense/Go2 depth → crop/resize `(87,58)` → same normalize as training; buffer index **`-2`**.
4. Enforce clip **±4.8**, `action_scale` **0.25**, PD **40/1**, 50 Hz.
5. Machine-side dry-run (log actions, no torque) → tethered flat ground.

Contract table (project docs): `ep-student-vs-rl-sar-contract.md`.
