#!/usr/bin/env bash
# Smoke go2_ep_student via RL_SAR_AUTO_RL (Passive->GetUp->RLLocomotion).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
XVFB_ROOT="${HOME}/.local/xvfb-root"
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
eval "$("${HOME}/.local/bin/micromamba" shell hook -s bash)"
micromamba activate rlsar
export PATH="${XVFB_ROOT}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${ROOT}/library/mujoco/lib:${ROOT}/library/inference_runtime/libtorch/lib:${XVFB_ROOT}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

pkill -f 'rl_sim_mujoco' >/dev/null 2>&1 || true
pkill -f 'Xvfb :99' >/dev/null 2>&1 || true
sleep 1
Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb99.log 2>&1 &
sleep 1
export DISPLAY=:99
export RL_SAR_GO2_CONFIG="${RL_SAR_GO2_CONFIG:-go2_ep_student}"
export RL_SAR_AUTO_RL=1

LOG=/tmp/smoke_ep_fsm.log
: > "$LOG"
cd "$ROOT"
timeout 55 ./cmake_build/bin/rl_sim_mujoco go2 go2 >>"$LOG" 2>&1 || true

# strip ANSI for matching
PLAIN=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" | tr '\r' '\n')
echo '==== KEY LINES ===='
echo "$PLAIN" | grep -E 'go2_ep_student|InitRL|RL Controller|ERROR|Exception|Failed|dual-input|ZERO depth|RLFSMStateRL|Successfully loaded' | grep -vE 'Getting up|Pre Getting' | head -40

if echo "$PLAIN" | grep -q 'InitRL() failed'; then
  echo 'SMOKE_FAIL: InitRL failed'; exit 1
fi
if echo "$PLAIN" | grep -q 'RL Controller \[go2_ep_student\]'; then
  echo 'SMOKE_OK: RL Controller go2_ep_student'; exit 0
fi
if echo "$PLAIN" | grep -q 'Successfully loaded Torch model'; then
  echo 'SMOKE_OK: EP model loaded into RLLocomotion'; exit 0
fi
echo 'SMOKE_FAIL: never reached RLLocomotion'; exit 2
