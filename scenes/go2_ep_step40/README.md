# Go2 EP 40cm step scenes (MuJoCo)

These XMLs are used by the plane-depth closed-loop contact tune.

- `scene_step40.xml` — stock Go2 friction (expects `go2.xml` + `assets/` from `src/rl_sar_zoo/go2_description/mjcf/`)
- `go2_fri15.xml` / `scene_step40_fri15.xml` — foot/floor/step sliding friction 1.5 (Isaac-domain)

Runtime eval (A6000): `/home/yihan/agent_ws/experiments/mujoco_contact_tune/mujoco_step40_eval.py --config fri15 --grid`

Do not set `EP_DEPTH_METRIC=range` as default.

## stable_v2 (2026-10-27)

On top of fri15: `solref=(0.005,1)` on feet/floor/step, XML `timestep=0.002`.
Policy loop must use `decimation=10` to keep 50 Hz. See `go2_contact_stable_v2.patch.md`.
Eval: `/home/yihan/agent_ws/experiments/mujoco_contact_tune/mujoco_step40_eval.py --config stable_v2`.
