#!/usr/bin/env bash
# Full-vision EP Student sim (recommended MXD path) via Isaac Gym sidecar.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source /home/yihan/extreme-parkour/activate.sh
cd "$ROOT"
exec python policy/go2/go2_ep_student/sim_isaac_bridge.py --backend jit --terrain parkour --steps "${STEPS:-1500}" --num-envs "${NUM_ENVS:-192}" "$@"
