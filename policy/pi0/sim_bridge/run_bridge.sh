#!/bin/bash
# Launch the RoboTwin Sim Bridge.
# Run on the SIM HOST (has SAPIEN/RoboTwin venv). Defaults: data 30001, control 30002.
#
# Usage:
#   ./run_bridge.sh                          # real SAPIEN env, task beat_block_hammer
#   ./run_bridge.sh --stub                   # StubSimEnv (no SAPIEN deps; x86 demo)
#   ./run_bridge.sh --task blocks_ranking_rgb --instruction "..."
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Same as script/eval_policy.py: curobo's torch.compile workers segfault on
# the cu124 image's Python 3.11.0rc1 (PEP 659 bug); the bridge never needs
# dynamo. Override with TORCHDYNAMO_DISABLE=0.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
# Default is relative to THIS script (policy/pi0/sim_bridge -> repo root), so
# the bridge launches correctly from any cwd. Canonicalize to an ABSOLUTE path
# before the cd below: sim_bridge.py receives --robotwin_root and chdirs into
# it again, so a relative value would be re-resolved against the new cwd and
# point outside the repo (causing "No module named 'eval_policy'" on reset).
ROBOTWIN_ROOT="$(cd "${ROBOTWIN_ROOT:-$SCRIPT_DIR/../../..}" && pwd)"

# RoboTwin env code reads ./assets relative to the repo root, so always run
# with cwd == ROBOTWIN_ROOT. Prefer the project venv next to this script
# (e.g. policy/pi0/.venv), then ROBOTWIN_ROOT/.venv.
PYTHON="python3"
if [ "$1" != "--stub" ]; then
  for VENV in "$SCRIPT_DIR/../.venv" "$ROBOTWIN_ROOT/.venv"; do
    if [ -f "$VENV/bin/activate" ]; then
      # shellcheck disable=SC1091
      source "$VENV/bin/activate"
      PYTHON="python"
      break
    fi
  done
fi

# Ensure protobuf runtime is available.
if ! python3 -c "import google.protobuf" >/dev/null 2>&1; then
  echo "[bridge] installing protobuf runtime..." >&2
  pip install -q protobuf
fi

cd "$ROBOTWIN_ROOT"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
exec "$PYTHON" "$SCRIPT_DIR/sim_bridge.py" --robotwin_root "$ROBOTWIN_ROOT" "$@"
