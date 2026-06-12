#!/usr/bin/env python3
"""Offline pi0.5 PyTorch inference from save_frame recordings.

Loads raw obs saved under eval video directories (episode*/frame_*), runs the
same input transforms -> torch model -> output transforms path as policy.py,
and writes environment-ready actions (no socket / C++ / filtering).

Run from the RoboTwin repository root:

    python policy/pi0/scripts/infer_from_frames.py
    python policy/pi0/scripts/infer_from_frames.py --config policy/pi0/infer_config.yaml
    python policy/pi0/scripts/infer_from_frames.py --overrides --frames_dir eval_result/... --output_dir /tmp/out
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import types
from typing import Any

# Suppress torch.compile / inductor autotune spam unless explicitly enabled.
os.environ.setdefault("TORCH_LOGS", "-dynamo,-inductor")

import jax
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

IMAGE_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEFAULT_INFER_CONFIG = pathlib.Path(__file__).resolve().parent.parent / "infer_config.yaml"


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    merged = dict(cfg)
    it = iter(overrides)
    for key in it:
        val = next(it)
        key = key.lstrip("-")
        if isinstance(val, str):
            lowered = val.lower()
            if lowered in {"true", "false"}:
                val = lowered == "true"
            elif val.isdigit():
                val = int(val)
            elif val.replace(".", "", 1).isdigit():
                val = float(val)
        merged[key] = val
    return merged


def _require(cfg: dict[str, Any], key: str) -> Any:
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"Missing required config field: {key}")
    return value


def load_frame_obs(frame_dir: pathlib.Path, *, state_dim: int = 14) -> dict[str, Any]:
    """Load one saved frame directory back into the raw policy obs dict."""
    images: dict[str, np.ndarray] = {}
    for idx, key in enumerate(IMAGE_KEYS):
        img_path = frame_dir / f"image_{idx}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        arr = np.asarray(Image.open(img_path))
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC image at {img_path}, got shape {arr.shape}")
        images[key] = np.transpose(arr, (2, 0, 1)).astype(np.uint8)

    prompt_path = frame_dir / "prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")

    state_path = frame_dir / "state.bin"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state: {state_path}")
    state = np.frombuffer(state_path.read_bytes(), dtype=np.float64)
    if state.size != state_dim:
        raise ValueError(
            f"Unexpected state size in {state_path}: got {state.size}, expected {state_dim}"
        )

    return {
        "images": images,
        "state": state,
        "prompt": prompt,
    }


def list_frame_dirs(episode_dir: pathlib.Path) -> list[pathlib.Path]:
    frame_dirs = sorted(episode_dir.glob("frame_*"))
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* directories found under {episode_dir}")
    return frame_dirs


def list_episode_dirs(frames_root: pathlib.Path) -> list[pathlib.Path]:
    episode_dirs = sorted(
        p for p in frames_root.iterdir() if p.is_dir() and p.name.startswith("episode")
    )
    if episode_dirs:
        return episode_dirs

    if list(frames_root.glob("frame_*")):
        return [frames_root]

    raise FileNotFoundError(
        f"No episode*/frame_* directories found under {frames_root}"
    )


def make_runtime_cfg(infer_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a policy.py-compatible cfg dict with non-torch paths disabled."""
    return {
        "stage": 1,  # SKIP: no socket / remote inference
        "use_cpp": False,
        "do_preproc": bool(infer_cfg.get("preproc", False)),
        "do_postproc": bool(infer_cfg.get("postproc", False)),
        "filter": 0,
        "save_frame": False,
        "debug": False,
        "port": 0,
        "visp": False,
        "chunk": int(infer_cfg.get("pi0_step", 50)),
        "fs": 50,
        "cutoff": 3,
        "channels": 14,
    }


def _resolve_sample_actions(
    model: torch.nn.Module,
    policy_sample_actions: Any,
    *,
    use_torch_compile: bool,
) -> Any:
    """Use uncompiled sample_actions by default for offline replay."""
    if use_torch_compile:
        return policy_sample_actions
    return types.MethodType(PI0Pytorch.sample_actions, model)


@dataclasses.dataclass
class Pi0TorchFrameInferencer:
    """Minimal pi0.5 torch inference pipeline extracted from Policy.infer()."""

    model: torch.nn.Module
    input_transform: Any
    output_transform: Any
    sample_actions: Any
    device: str
    sample_kwargs: dict[str, Any]
    pi0_step: int

    @classmethod
    def from_checkpoint(
        cls,
        *,
        train_config_name: str,
        model_path: str | pathlib.Path,
        robotwin_repo_id: str,
        infer_cfg: dict[str, Any],
        device: str | None = None,
    ) -> Pi0TorchFrameInferencer:
        runtime_cfg = make_runtime_cfg(infer_cfg)
        train_config = _config.get_config(train_config_name)
        policy = _policy_config.create_trained_policy(
            train_config,
            model_path,
            robotwin_repo_id=robotwin_repo_id,
            cfg=runtime_cfg,
            pytorch_device=device,
        )
        if policy._model is None:
            raise RuntimeError("Torch model was not loaded.")
        if not policy._is_pytorch_model:
            raise RuntimeError("Expected a PyTorch checkpoint (model.safetensors).")

        pi0_step = int(infer_cfg.get("pi0_step", runtime_cfg["chunk"]))
        use_torch_compile = bool(infer_cfg.get("torch_compile", False))
        sample_actions = _resolve_sample_actions(
            policy._model,
            policy._sample_actions,
            use_torch_compile=use_torch_compile,
        )
        if not use_torch_compile:
            print("Using uncompiled sample_actions (torch_compile=false).")

        return cls(
            model=policy._model,
            input_transform=policy._input_transform,
            output_transform=policy._output_transform,
            sample_actions=sample_actions,
            device=policy._pytorch_device,
            sample_kwargs=dict(policy._sample_kwargs),
            pi0_step=pi0_step,
        )

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        """Run full post-processed action chunk for one raw observation."""
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self.input_transform(inputs)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self.device)[None, ...],
            inputs,
        )

        _, action = self.sample_actions(
            self.device,
            _model.Observation.from_dict(inputs),
            **self.sample_kwargs,
        )

        outputs = {
            "state": inputs["state"],
            "actions": action,
        }
        outputs = jax.tree.map(
            lambda x: np.asarray(x[0, ...].detach().cpu()),
            outputs,
        )
        outputs = self.output_transform(outputs)
        return np.asarray(outputs["actions"])

    def infer_env_actions(self, obs: dict[str, Any]) -> np.ndarray:
        """Return the first pi0_step actions, matching deploy_policy eval()."""
        actions = self.infer(obs)
        return actions[: self.pi0_step]


def save_actions(
    output_dir: pathlib.Path,
    frame_name: str,
    actions: np.ndarray,
    env_actions: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{frame_name}_actions.npy", actions)
    np.save(output_dir / f"{frame_name}_env_actions.npy", env_actions)


def run_episode(
    inferencer: Pi0TorchFrameInferencer,
    episode_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    state_dim: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    frame_dirs = list_frame_dirs(episode_dir)
    episode_name = episode_dir.name if episode_dir.name.startswith("episode") else "episode0"
    episode_output = output_dir / episode_name
    episode_output.mkdir(parents=True, exist_ok=True)

    for frame_dir in tqdm(frame_dirs, desc=episode_name, leave=False):
        obs = load_frame_obs(frame_dir, state_dim=state_dim)
        actions = inferencer.infer(obs)
        env_actions = actions[: inferencer.pi0_step]
        save_actions(episode_output, frame_dir.name, actions, env_actions)
        tqdm.write(f"{episode_name}/{frame_dir.name}: saved actions {actions.shape}")
        results.append(
            {
                "episode": episode_name,
                "frame": frame_dir.name,
                "actions_shape": list(actions.shape),
                "env_actions_shape": list(env_actions.shape),
                "first_env_action": env_actions[0].tolist(),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pi0.5 torch inference on save_frame recordings."
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_INFER_CONFIG,
        help="Path to infer_config.yaml.",
    )
    parser.add_argument(
        "--overrides",
        nargs=argparse.REMAINDER,
        help="Override config values, e.g. --overrides --frames_dir path --output_dir /tmp/out",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer_cfg = _load_yaml(args.config)
    if args.overrides:
        infer_cfg = _apply_overrides(infer_cfg, args.overrides)

    frames_dir = pathlib.Path(_require(infer_cfg, "frames_dir"))
    output_dir = pathlib.Path(_require(infer_cfg, "output_dir"))
    train_config_name = str(_require(infer_cfg, "train_config_name"))
    model_path = _require(infer_cfg, "model_path")
    robotwin_repo_id = infer_cfg.get("robotwin_repo_id")
    if robotwin_repo_id is None:
        raise ValueError("Missing required config field: robotwin_repo_id")

    inferencer = Pi0TorchFrameInferencer.from_checkpoint(
        train_config_name=train_config_name,
        model_path=model_path,
        robotwin_repo_id=str(robotwin_repo_id),
        infer_cfg=infer_cfg,
        device=infer_cfg.get("device"),
    )

    episode = infer_cfg.get("episode")
    if episode is not None:
        episode_dirs = [frames_dir / str(episode)]
        if not episode_dirs[0].is_dir():
            raise FileNotFoundError(f"Episode directory not found: {episode_dirs[0]}")
    else:
        episode_dirs = list_episode_dirs(frames_dir)

    state_dim = int(infer_cfg.get("state_dim", 14))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        summary.extend(
            run_episode(
                inferencer,
                episode_dir,
                output_dir,
                state_dim=state_dim,
            )
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved actions for {len(summary)} frames to {output_dir}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
