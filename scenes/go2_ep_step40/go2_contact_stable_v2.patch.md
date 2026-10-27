# Recommended MuJoCo contact/dynamics overlay (stable_v2)

Relative to stock `go2.xml` / `scene_step40.xml`, and on top of fri15:

1. Foot / floor / step sliding friction → `1.5 0.005 0.0001` (same as fri15)
2. Foot / floor / step `solref` → `(0.005, 1)` (stiffer than MuJoCo default 0.02)
3. `timestep` → `0.002`, policy `decimation` → `10` (keep 50 Hz control; finer contact integrate)
4. Keep armature=0.01, margin=0.001, solimp default, impratio=100, Kp=40, Kd=1, action_scale=0.25
5. Plane depth contract unchanged

Rationale (2026-10-27 v2): vs fri15 on cmd 0.8–1.3 @3/3, stable_v2 raises
post-climb hold (~+46 steps), lowers on-deck roll/pitch RMS and body angvel,
keeps full pass grid, and extends robust window through cmd 0.7 and 1.4.
