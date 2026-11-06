# Weight pointers (cs2)

Symlink these into this directory (or copy):

```bash
EP_TRACED=/home/yihan/extreme-parkour/extreme-parkour-repro/legged_gym/logs/parkour_new/922-10-GO2-STUDENT-D0/traced
ln -sfn "$EP_TRACED/922-10-GO2-STUDENT-D0-59500-base_jit.pt" policy/go2/go2_ep_student/base_jit.pt
ln -sfn "$EP_TRACED/922-10-GO2-STUDENT-D0-59500-vision_weight.pt" policy/go2/go2_ep_student/vision_weight.pt
```
