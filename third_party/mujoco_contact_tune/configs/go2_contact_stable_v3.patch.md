# Recommended MuJoCo contact/dynamics overlay (stable_v3)

On top of stable_v2 (fri15 + solref 0.005 + dt 0.002/dec10):

1. Keep sliding friction = 1.5
2. Torsional/rolling friction → near-zero `(1.5, 1e-4, 1e-4)` (cut spin cone kicks)
3. Joint damping 0.1 → **0.6** (viscous joint damping to cut flail)
4. Keep armature=0.01, margin=0.001, solimp default, impratio=100, Kp=40, Kd=1, action_scale=0.25
5. Plane depth contract unchanged

Rationale (2026-10-27 v3): vs stable_v2 on cmd 0.8–1.3 @3/3, stable_v3 lowers
joint_vel_rms (~−3.7), calf_vel_rms (~−1.2), joint_vel_peak (~−12), keeps full pass
grid and post_climb_hold nearly flat. Closer to Isaac same-box joint-vel ballpark (~12).
