#!/bin/bash

gpu_id=${1:-0}
shift $(( $# > 0 ? 1 : 0 ))

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..
source policy/pi0/.venv/bin/activate

if [ $# -gt 0 ]; then
  PYTHONWARNINGS=ignore::UserWarning \
  python policy/pi0/scripts/infer_from_frames.py \
      --config policy/pi0/infer_config.yaml \
      --overrides "$@"
else
  PYTHONWARNINGS=ignore::UserWarning \
  python policy/pi0/scripts/infer_from_frames.py \
      --config policy/pi0/infer_config.yaml
fi
