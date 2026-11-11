# stable_isaac_v6 / damping_isaac_v6 / v6_sol_d09 (GO2-33)

On top of stable_isaac_v5 (ia_hip8):
- solref = (0.005, **0.9**)  # dampratio only; timeconst stays 0.005
- hip 0.8 / front calf 3.8 / front thigh 1.0 / rear calf 2.5 unchanged

Plane depth unchanged. Core 15/15, wide 21/21.

```
source /home/yihan/extreme-parkour/activate.sh
cd /home/yihan/agent_ws/experiments/damping_isaac_v6
python run_front_leg_ab.py --configs stable_isaac_v6 --cmds 0.8,1.0,1.1,1.2,1.3 --trials 3
```
