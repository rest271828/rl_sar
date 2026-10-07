#!/usr/bin/env bash
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
export RL_SAR_CMD_X="${RL_SAR_CMD_X:-0}"

LOG=/tmp/smoke_ep_fsm.log
: > "$LOG"
cd "$ROOT"
timeout 70 ./cmake_build/bin/rl_sim_mujoco go2 go2 >>"$LOG" 2>&1 || true

PLAIN=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" | tr '\r' '\n')
echo '==== KEY LINES ===='
echo "$PLAIN" | grep -E 'go2_ep_student|InitRL|RL Controller|ERROR|Successfully loaded|EP_MUJOCO|\[EP_MUJOCO\]|RLFSMStateRL' | grep -vE 'Getting up|Pre Getting' | head -60

if echo "$PLAIN" | grep -q 'InitRL() failed'; then echo 'SMOKE_FAIL: InitRL failed'; exit 1; fi
if ! echo "$PLAIN" | grep -q 'RL Controller \[go2_ep_student\]'; then echo 'SMOKE_FAIL: never reached RL'; exit 2; fi

# Behavior summary from EP_MUJOCO|\[EP_MUJOCO\] height samples
python3 - <<'PY'
import re
from pathlib import Path
t=Path('/tmp/smoke_ep_fsm.log').read_text(errors='ignore')
t=re.sub(r'\x1b\[[0-9;]*m','',t)
zs=[]; amaxs=[]
for m in re.finditer(r'EP_MUJOCO|\[EP_MUJOCO\] t=(\d+) z=([-\d.]+) cmd_x=([-\d.]+) amax=([-\d.]+)', t):
    zs.append(float(m.group(2))); amaxs.append(float(m.group(4)))
if not zs:
    print('SMOKE_PARTIAL: RL ok but no height samples')
else:
    z_mean=sum(zs)/len(zs); z_min=min(zs); z_max=max(zs)
    a_mean=sum(amaxs)/len(amaxs)
    print(f'EP_BEHAVIOR samples={len(zs)} z_mean={z_mean:.3f} z_min={z_min:.3f} z_max={z_max:.3f} amax_mean={a_mean:.3f}')
    # Go2 stand height target ~0.30; spawn 0.42-ish in mujoco; collapsed <0.15
    if z_mean >= 0.18 and z_min >= 0.12:
        print('SMOKE_OK: go2_ep_student running; base height looks non-collapsed')
    else:
        print('SMOKE_WARN: RL runs but base height low (zero depth latent limits locomotion)')
PY
