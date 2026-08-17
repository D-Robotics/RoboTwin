#!/usr/bin/env python3
"""Headless smoke test for the REAL SimEnv wrapper (no board/browser needed).

Validates that env_wrapper.SimEnv can drive a real RoboTwin TASK_ENV:
list_tasks -> reset -> get_obs (3 raw-res images + state) -> take_action ->
close. Images are sent at RAW camera resolution (e.g. 640x480); the board
letterboxes to 224x224 like the eval.sh path.

Run from the RoboTwin repo root with the RoboTwin python:

  cd /mnt/data/yanjie.shen/RoboTwin
  conda activate RoboTwin            # or your sapien env
  pip install protobuf -q           # only if missing
  export PYTHONPATH=<repo>/llm_engine/demo/vla_demo/sim_bridge:$PYTHONPATH
  python <repo>/llm_engine/demo/vla_demo/sim_bridge/smoke_test_env.py \
      --robotwin_root /mnt/data/yanjie.shen/RoboTwin --task beat_block_hammer
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from env_wrapper import SimEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robotwin_root", required=True)
    ap.add_argument("--task", default="beat_block_hammer")
    ap.add_argument("--task_config", default="demo_clean")
    args = ap.parse_args()

    os.chdir(args.robotwin_root)
    sys.path.insert(0, args.robotwin_root)
    sys.path.insert(0, os.path.join(args.robotwin_root, "script"))

    env = SimEnv(args.robotwin_root, task_config=args.task_config)
    print(f"[smoke] tasks: {env.list_tasks()[:8]} ... ({len(env.list_tasks())} total)")
    print(f"[smoke] reset task={args.task}")
    env.reset(task=args.task, seed=0, instruction="")
    print(f"[smoke] instruction: {env.instruction!r}")
    print(f"[smoke] step_lim={env.step_lim} take_action_cnt={env.take_action_cnt}")

    rgb, state, instr = env.get_obs()
    print(f"[smoke] obs: {len(rgb)} imgs, shapes={[im.shape for im in rgb]}, "
          f"dtype={rgb[0].dtype}; state len={len(state)} dtype={state.dtype}")
    assert len(rgb) == 3, "expected 3 camera images"
    # Raw camera resolution: HxWx3 uint8, any H/W (board letterboxes to 224).
    assert rgb[0].ndim == 3 and rgb[0].shape[2] == 3, f"not HxWx3: {rgb[0].shape}"
    assert rgb[0].dtype == np.uint8
    assert len(state) == 14, f"state dim != 14: {len(state)}"

    action = np.zeros(14, dtype=np.float64)
    env.take_action(action)
    print(f"[smoke] take_action OK; take_action_cnt={env.take_action_cnt} "
          f"eval_success={env.eval_success()}")

    env.close()
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
