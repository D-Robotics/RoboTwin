#!/bin/bash
# Launch the mock board server (x86 dev loop, no board/HBM needed).
# Serves vla.html + REST + SSE on :8080 and acts as protobuf client to the Sim Bridge.
#
# Usage:
#   ./run_mock_board.sh                         # default pi0_step=50, state_dim=14
#   ./run_mock_board.sh --port 8080
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/sim_bridge:${PYTHONPATH:-}"
exec python3 "$SCRIPT_DIR/mock_board_server.py" "$@"
