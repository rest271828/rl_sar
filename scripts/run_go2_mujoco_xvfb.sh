#!/usr/bin/env bash
# Run rl_sar MuJoCo Go2 under a user-space Xvfb (no root/display required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
XVFB_ROOT="${HOME}/.local/xvfb-root"
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
# shellcheck disable=SC1091
eval "$("${HOME}/.local/bin/micromamba" shell hook -s bash)"
micromamba activate rlsar

export PATH="${XVFB_ROOT}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${ROOT}/library/mujoco/lib:${ROOT}/library/inference_runtime/libtorch/lib:${XVFB_ROOT}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

CONFIG="${RL_SAR_GO2_CONFIG:-himloco}"
# Optional: RL_SAR_AUTO_RL=1 skips Passive/GetUp keys and enters RLLocomotion
export RL_SAR_GO2_CONFIG="$CONFIG"

if ! pgrep -f 'Xvfb :99' >/dev/null 2>&1; then
  Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb99.log 2>&1 &
  sleep 1
fi
export DISPLAY=:99

echo "[run_go2_mujoco_xvfb] RL_SAR_GO2_CONFIG=${RL_SAR_GO2_CONFIG}"
echo "[run_go2_mujoco_xvfb] binary=${ROOT}/cmake_build/bin/rl_sim_mujoco"
exec "${ROOT}/cmake_build/bin/rl_sim_mujoco" go2 go2
