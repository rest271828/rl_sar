# Recommended MuJoCo contact overlay (fri15)

Change relative to stock `go2.xml` / `scene_step40.xml`:

1. Foot geom class friction: `0.4 0.02 0.01` → `1.5 0.005 0.0001`
2. Floor + step40 sliding friction → `1.5 0.005 0.0001`
3. Keep armature=0.01, margin=0.001, solref/solimp defaults, impratio=100, timestep=0.005

Rationale: Isaac `static/dynamic_friction=1.0` with domain-rand `[0.6, 2.0]`.
Stock MuJoCo foot fri=0.4 is below the training support. fri=1.5 sits in the
upper-mid of the Isaac rand range and yields 5/5 speed-grid passes (3/3 each).
