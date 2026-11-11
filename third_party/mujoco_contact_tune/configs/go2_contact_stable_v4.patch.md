# Recommended MuJoCo contact/dynamics overlay (stable_v4)

On top of stable_v3 (damp0.6 + spin0 + solref 0.005 + dt 0.002/dec10):

1. Keep sliding friction = 1.5; torsional/rolling near-zero
2. Keep hip/thigh joint damping = 0.6
3. **Calf/knee joint damping → 2.5** (calf-only viscous; main v4 lever)
4. Keep armature=0.01, margin=0.001, solimp default, impratio=100, Kp=40, Kd=1, action_scale=0.25
5. Plane depth contract unchanged

Rationale (2026-10-28 v4): vs stable_v3 on cmd 0.8–1.3 @3/3, stable_v4 lowers
calf_vel_rms (~4.56→~2.80, Isaac≈2.94) and joint_vel_rms (~14.5→~11.0) while keeping
full core pass grid. Alias `sv3_calf25`.
