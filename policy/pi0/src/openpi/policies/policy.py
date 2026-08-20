from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

import torch
import os
from PIL import Image

from openpi.patch.filters import NoFilter, FIR, ZeroPhaseFTR, MultiChannelButterworth
from openpi.patch.utils import dict_equal
from openpi.policies.eval_progress import EvalLiveProgress, eval_live
from openpi.policies.remote_model import RemoteModel

BasePolicy: TypeAlias = _base_policy.BasePolicy

OBS, SKIP, FULL = range(3)
DBG = False

_RESET = "\033[0m"

_BOLD_MAGENTA = "\033[1;35m"
_BOLD_GREEN = "\033[1;32m"
_BOLD_RED = "\033[1;31m"
_BOLD_YELLOW_UL = "\033[1;4;33m"

_BORDER = "━" * 80


def print_startup_banner(*, trailing_blank: bool = False) -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                          PI0 DEMO                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    if trailing_blank:
        print()


def _flush_progress_line() -> None:
    print("\r\033[K", end="", flush=True)


def print_episode_start(episode_id: int, prompt: str) -> None:
    _flush_progress_line()
    print()
    print(f"{_BOLD_MAGENTA}[EPISODE START]{_RESET}")
    print(_BORDER)
    print(f"  Actor  : {episode_id}")
    print(f"  {_BOLD_YELLOW_UL}Prompt : {prompt}{_RESET}")
    print()


def print_episode_section(episode_id: int, prompt: str) -> None:
    print_episode_start(episode_id, prompt)


def print_episode_end(
    *,
    success: bool,
    step: int,
    step_lim: int,
    task_name: str,
    policy_name: str,
    task_config: str = "",
    ckpt_setting: str = "",
    suc: int,
    test_num: int,
    seed: int,
) -> None:
    _flush_progress_line()
    success_rate = round(suc / test_num * 100, 1) if test_num else 0.0
    result_tag = f"{_BOLD_GREEN}[EPISODE SUCCESS]{_RESET}" if success else f"{_BOLD_RED}[EPISODE FAIL]{_RESET}"
    print(_BORDER)
    print(f"{result_tag} (Step: {step} / {step_lim})")
    print(
        f"  \033[1mSuccess Rate:\033[0m \033[96m{suc}/{test_num}\033[0m "
        f"(\033[95m{success_rate}%\033[0m) | "
        f"\033[93m{task_name}\033[0m | \033[94m{policy_name}\033[0m | "
        f"seed: \033[90m{seed}\033[0m"
    )
    print(_BORDER)
    print()


# SKIP: Robotwin raw procedure
# OBS: Send and receive obs to test consistency
# FULL: Send obs and receive aloha action result(w/ postproc)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        cfg=None,
    ):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        if model is not None:
            if self._is_pytorch_model:
                self._model = self._model.to(pytorch_device)
                self._model.eval()
                self._sample_actions = model.sample_actions
            else:
                # JAX model setup
                self._sample_actions = nnx_utils.module_jit(model.sample_actions)
                self._rng = rng or jax.random.key(0)
        else:
            self._is_pytorch_model=True
        self._io_log_cycles = 0

        # config
        self.stage = cfg["stage"]
        if self.stage == FULL:
            eval_live.enabled = True
        self.port = cfg["port"]
        self.do_preproc = cfg["do_preproc"] and self.stage == FULL
        self.do_postproc = cfg["do_postproc"] and self.stage == FULL
        self.visp = cfg["visp"]
        self.chunk = cfg["chunk"]
        self.debug = cfg["debug"]
        self.save_frame = cfg.get("save_frame", False)
        self.save_all_frames = cfg.get("save_all_frames", False)
        self.frame_save_dir = cfg.get("eval_video_save_dir")
        self._frame_episode_idx = 0
        self._frame_idx = 0
        self._frame_episode_dir: pathlib.Path | None = None
        self._frame_save_started = False
        if self.debug:
            if not os.path.exists("test"):
                os.mkdir("test")
                
        # filter
        self.filter_type = cfg["filter"]
        filter_keys = ["fs", "cutoff", "channels"]
        common_params = {key: cfg.get(key) for key in filter_keys if key in cfg}

        filter_zoo = [NoFilter, MultiChannelButterworth, FIR, ZeroPhaseFTR]
        filter_class = filter_zoo[self.filter_type]
        filter_names = [
            "No filter loaded.",
            "Multi-channel Butterworth filter loaded.",
            "FIR filter loaded.",
            "Zero-phase FIR filter loaded.",
        ]

        self.filter = filter_class(**common_params)
        self.filter_py = filter_class(**common_params)
        print(filter_names[self.filter_type])

        self.remote = RemoteModel(
            port=self.port,
            do_preproc=self.do_preproc,
            do_postproc=self.do_postproc,
            visp=self.visp,
            chunk=self.chunk,
            debug=self.debug,
            is_pytorch_model=self._is_pytorch_model,
        )

    def filt(self, arr):
        return self.filter.filter(arr)

    def filt_py(self, arr):
        return self.filter_py.filter(arr)

    def reset_filter(self):
        self.filter.reset()
        self.filter_py.reset()

    def connect(self) -> None:
        self.remote.connect()

    def disconnect(self) -> None:
        self.remote.disconnect()

    def preprocess_inputs(self,inputs):
        obs = _model.Observation.from_dict(inputs)
        obs = _preprocessing.preprocess_observation_pytorch(obs, train=False)
        obs = {"images": obs.images, "state": obs.state.to(torch.float32), "prompt": obs.tokenized_prompt}
        return obs

    def _to_numpy(self, value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _save_image_jpg(self, image: Any, path: pathlib.Path) -> None:
        arr = self._to_numpy(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path, format="JPEG")

    def _save_raw_obs(self, obs: dict, frame_dir: pathlib.Path) -> None:
        frame_dir.mkdir(parents=True, exist_ok=True)

        image_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        images = obs.get("images", {})
        for idx, key in enumerate(image_keys):
            if key in images:
                self._save_image_jpg(images[key], frame_dir / f"image_{idx}.jpg")

        prompt = obs.get("prompt", "")
        if isinstance(prompt, (bytes, bytearray)):
            prompt = prompt.decode("utf-8")
        elif not isinstance(prompt, str):
            prompt = str(prompt)
        (frame_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        state = self._to_numpy(obs["state"]).astype(np.float64, copy=False)
        (frame_dir / "state.bin").write_bytes(state.tobytes())

    def save_obs(self, obs: dict, reset: bool = False) -> None:
        """Save raw obs to the eval video directory when save_frame is enabled."""
        if not self.save_frame or not self.frame_save_dir:
            return

        if reset:
            self._frame_idx = 0
            if self._frame_save_started:
                self._frame_episode_idx += 1
            else:
                self._frame_episode_idx = 0
                self._frame_save_started = True
            self._frame_episode_dir = pathlib.Path(self.frame_save_dir) / f"episode{self._frame_episode_idx}"
            self._frame_episode_dir.mkdir(parents=True, exist_ok=True)

        if self._frame_episode_dir is None:
            self._frame_episode_dir = pathlib.Path(self.frame_save_dir) / f"episode{self._frame_episode_idx}"
            self._frame_episode_dir.mkdir(parents=True, exist_ok=True)

        frame_dir = self._frame_episode_dir / f"frame_{self._frame_idx:06d}"
        self._save_raw_obs(obs, frame_dir)
        self._frame_idx += 1

    @override
    def infer(
        self,
        obs: dict,
        reset=False,
        noise: np.ndarray | None = None,
        env_step: int | None = None,
        env_step_lim: int | None = None,
    ) -> dict:  # type: ignore[misc]
        if self.stage != SKIP:
            self.connect()

        if reset:
            self.reset_filter()
            self._io_log_cycles = 0
            eval_live.reset_episode()

        if self.save_frame and not self.save_all_frames:
            self.save_obs(obs, reset)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        
        # Input Process For torch model
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Preprocess inputs
        preproc_inputs = self.preprocess_inputs(inputs)

        sent_obs = preproc_inputs if self.do_preproc else obs
        action_recv = None
        
        # Connect Mode
        if self.stage == FULL:
            cycle = self._io_log_cycles
            verbose_io = self.debug or cycle == 0
            if verbose_io:
                action_recv = self.remote(sent_obs, reset=reset, verbose=True)
                eval_live.complete_cycle_verbose(cycle, env_step, env_step_lim)
            else:
                eval_live.begin_infer_cycle(cycle, env_step, env_step_lim)
                action_recv = self.remote(sent_obs, reset=reset, verbose=False, live_io=eval_live)
            self._io_log_cycles += 1
            # Just Return if no model loaded
            if self._model is None:
                outputs = {"actions": action_recv, "state": preproc_inputs["state"]}
                outputs = jax.tree.map(lambda x: np.asarray(x.squeeze().detach().cpu()), outputs)
                if self.do_postproc:
                    outputs = self._output_transform(outputs)
                outputs["actions"] = self.filt(outputs["actions"])
                return outputs

        # Test Mode
        if self.stage == OBS:
            self.remote.send(obs)
            recv_data = self.remote.receive()
            obs_old = obs
            obs = self.remote.proc_recv(recv_data)
            assert dict_equal(obs, obs_old), "recv mismatch!"
            
        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise


        _, action_local = self._sample_actions(
            sample_rng_or_pytorch_device, _model.Observation.from_dict(inputs), **self._sample_kwargs
        )

        local_outputs = {
            "state": inputs["state"],
            "actions": action_local,
        }
        
        if self._is_pytorch_model:
            outputs_mapped = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), local_outputs)
        else:
            outputs_mapped = jax.tree.map(lambda x: np.asarray(x[0, ...]), local_outputs)
            
        # Compare action_recv & local_action
        if action_recv is not None:
            action_recv = action_recv.squeeze()
        if self.do_postproc:
            if self.debug:
                print("raw action", outputs_mapped["actions"])
                print("cpp action", action_recv)
                np.save("test/py_act.npy", np.array(outputs_mapped["actions"]))
                np.save("test/cpp_act.npy", np.array(action_recv))
            outputs_mapped['actions'] = action_recv
        
        # Transform action
        outputs = self._output_transform(outputs_mapped)
        action_raw = outputs["actions"]

        # Compare action_recv & transformed action
        if self.stage == FULL:
            if not self.do_postproc:
                outputs["actions"] = action_recv
                if self.debug:
                    print(action_raw)
                    print(action_recv)
                    np.save("test/py_act.npy", np.array(action_raw))
                    np.save("test/cpp_act.npy", np.array(action_recv))

        # Filter
        outputs["actions"] = self.filt(outputs["actions"])
        if self.debug:
            filted_py = self.filt_py(action_raw)
            np.save("test/py_filt.npy", filted_py)
            np.save("test/cpp_filt.npy", outputs["actions"])

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
