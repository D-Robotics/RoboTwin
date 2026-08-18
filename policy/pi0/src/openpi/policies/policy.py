from collections.abc import Sequence
import logging
import pathlib
import struct
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

import socket
import select
import time

from safetensors.torch import load_file
import torch.nn as nn
import torch
import einops
import os
from PIL import Image

from openpi.patch import msg_pb2
from openpi.patch.filters import NoFilter, FIR, ZeroPhaseFTR, MultiChannelButterworth
from openpi.patch.utils import dict_equal
from openpi.policies.eval_progress import EvalLiveProgress, eval_live

BasePolicy: TypeAlias = _base_policy.BasePolicy

OBS, SKIP, FULL = range(3)
DBG = False

_BLUE = "\033[34m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_GRAY = "\033[90m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

_BOLD_MAGENTA = "\033[1;35m"
_BOLD_GREEN = "\033[1;32m"
_BOLD_RED = "\033[1;31m"
_BOLD_YELLOW_UL = "\033[1;4;33m"

_MAGENTA = "\033[35m"

_BORDER = "━" * 80

_DTYPE_NAMES = {
    msg_pb2.Tensor.FLOAT64: "float64",
    msg_pb2.Tensor.UINT8: "uint8",
    msg_pb2.Tensor.STRING: "string",
    msg_pb2.Tensor.FLOAT32: "float32",
    msg_pb2.Tensor.INT32: "int32",
    msg_pb2.Tensor.FP16: "float16",
}


def _dtype_name(dtype: int) -> str:
    return _DTYPE_NAMES.get(dtype, str(dtype))


def _shape_str(shape) -> str:
    if not shape:
        return "scalar"
    return "x".join(str(dim) for dim in shape)


def print_startup_banner(*, trailing_blank: bool = False) -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                          PI0 DEMO                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    if trailing_blank:
        print()


def _flush_progress_line() -> None:
    print("\r\033[K", end="", flush=True)


def _format_header_inline(header, *, show_reset: bool = True) -> str:
    parts = [f"Seq: {header.seq}"]
    if show_reset:
        reset_str = str(header.reset).lower()
        if header.reset:
            parts.append(f"{_YELLOW}Reset: true{_RESET}")
        else:
            parts.append(f"Reset: {reset_str}")
    return " | ".join(parts)


def _status_text(ok: bool) -> str:
    if ok:
        return f"{_GREEN}OK{_RESET}"
    return f"{_RED}FAILED{_RESET}"


def _format_body_lines(batch) -> list[str]:
    lines: list[str] = []
    if batch.images:
        img = batch.images[0]
        lines.append(
            f"Images: {len(batch.images)} ({_dtype_name(img.dtype)}, {_shape_str(img.shape)})"
        )
    if batch.languages:
        lang = batch.languages[0]
        lines.append(
            f"Languages: {len(batch.languages)} ({_dtype_name(lang.dtype)}, {_shape_str(lang.shape)})"
        )
    if batch.states:
        state = batch.states[0]
        lines.append(
            f"States: {len(batch.states)} ({_dtype_name(state.dtype)}, {_shape_str(state.shape)})"
        )
    return lines


def _io_tag_label(tag: str) -> str:
    if tag == "SEND":
        color = _BLUE
    elif tag == "RECEIVE":
        color = _MAGENTA
    else:
        color = ""
    if color:
        return f"{color}[{tag}]{_RESET}"
    return f"[{tag}]"


def _print_io_verbose(
    tag: str,
    ok: bool,
    payload_bytes: int,
    header=None,
    batch=None,
) -> None:
    _flush_progress_line()
    status = _status_text(ok)
    payload = f"{_GRAY}({payload_bytes} bytes){_RESET}"
    print(f"  {_io_tag_label(tag)} Status: {status} {payload}")
    if header is not None:
        print(f"    Header ➔ {_format_header_inline(header, show_reset=(tag == 'SEND'))}")
    if batch is not None:
        body_lines = _format_body_lines(batch)
        for idx, line in enumerate(body_lines):
            prefix = "    Body   ➔ " if idx == 0 else "           ➔ "
            print(f"{prefix}{line}")
    print()


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
        self.listen_fd = None
        self.sock_fd = None
        self.seq = 0
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
            if not os.path.exists("test/scp"):
                os.mkdir("test/scp")
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

    def filt(self, arr):
        return self.filter.filter(arr)

    def filt_py(self, arr):
        return self.filter_py.filter(arr)

    def reset_filter(self):
        self.filter.reset()
        self.filter_py.reset()

    def _close_socket(self, sock: socket.socket | None) -> None:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def disconnect(self) -> None:
        self._close_socket(self.sock_fd)
        self._close_socket(self.listen_fd)
        self.sock_fd = None
        self.listen_fd = None

    def _is_connected(self, sock: socket.socket | None) -> bool:
        if sock is None:
            return False
        try:
            sock.getpeername()
            return True
        except OSError:
            return False

    def connect(self):
        if self._is_connected(self.sock_fd):
            return

        self.disconnect()

        # 1. 创建 TCP Socket
        try:
            self.listen_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.listen_fd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except OSError as e:
            print(f"创建 Socket 失败：{str(e)}")
            return

        # 2. 绑定端口
        server_addr = ("0.0.0.0", self.port)
        try:
            self.listen_fd.bind(server_addr)
        except OSError as e:
            print(f"绑定端口失败：{str(e)}")
            self.disconnect()
            return

        # 3. 开始监听连接
        try:
            self.listen_fd.listen(5)
        except OSError as e:
            print(f"监听失败：{str(e)}")
            self.disconnect()
            return

        _flush_progress_line()
        print(f"{_BLUE}[CONNECT]{_RESET}")
        print(f"  Listen : 0.0.0.0:{self.port}")

        connect_start = time.time()
        try:
            while True:
                elapsed = int(time.time() - connect_start)
                readable, _, _ = select.select([self.listen_fd], [], [], 1.0)
                print(f"\r  Status : {_YELLOW}WAITING{_RESET} {elapsed}s", end="", flush=True)
                if readable:
                    self.sock_fd, client_addr = self.listen_fd.accept()
                    break
            print(f"\r\033[K  Client : {client_addr[0]}:{client_addr[1]}")
            print(f"  Status : {_GREEN}CONNECTED{_RESET}")
            print()
            print("Engine Starting...")
        except OSError as e:
            print()
            print(f"接受连接失败：{str(e)}")
            self.disconnect()
            return

        # 单客户端模式：accept 后关闭 listen fd，避免重连时泄漏
        self._close_socket(self.listen_fd)
        self.listen_fd = None

    def send(self, observation: dict, reset=False, verbose: bool = True, live_io: EvalLiveProgress | None = None) -> bool:
        sock = self.sock_fd
        """发送多模态输入消息"""
        input_msg = msg_pb2.MultiModalInput()

        def build_header(seq):
            """构建Header消息"""
            header = msg_pb2.Header()
            header.seq = seq

            def get_current_time():
                """获取当前时间(秒和纳秒)"""
                current = time.time()
                sec = int(current)
                nsec = int((current - sec) * 1e9)
                return sec, nsec

            sec, nsec = get_current_time()
            header.stamp.sec = sec
            header.stamp.nsec = nsec
            header.frame_id = "camera_optical_frame"
            header.reset = reset
            return header

        # 设置Header
        header = build_header(self.seq)
        self.seq += 1

        input_msg.header.CopyFrom(header)

        img_dtypes = [msg_pb2.Tensor.UINT8, msg_pb2.Tensor.FLOAT32]
        lang_dtypes = [msg_pb2.Tensor.STRING, msg_pb2.Tensor.INT32]
        state_dtypes = [msg_pb2.Tensor.FLOAT64, msg_pb2.Tensor.FLOAT32]

        # 添加图像Tensor
        if type(observation["images"]) is dict:
            for img in observation["images"].values():
                image = input_msg.images.add()
                image.dtype = img_dtypes[self.do_preproc]
                if self.do_preproc:
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu().numpy()
                    img = img.astype(np.float32)
                    image.dtype = msg_pb2.Tensor.FLOAT32
                image.shape.extend(img.shape)
                image.data = img.tobytes()
        else:
            # send kvcache
            img = observation["images"]
            image = input_msg.images.add()
            if self.visp:
                img = img.astype(jnp.float16)
                image.dtype = msg_pb2.Tensor.FP16
            else:
                img = img.astype(jnp.float32)
                image.dtype = msg_pb2.Tensor.FLOAT32
            image.shape.extend(img.shape)
            image.data = img.tobytes()

        # 添加语言Tensor
        language = input_msg.languages.add()
        language.dtype = lang_dtypes[self.do_preproc]
        language.shape.extend(observation["prompt"].shape)
        lang =  observation["prompt"]
        if isinstance(lang, torch.Tensor):
            lang = lang.detach().cpu().numpy()
        language.data = (
            lang.encode("utf-8")
            if not self.do_preproc
            else lang.astype(np.int32).tobytes()
        )

        # 添加状态Tensor
        state = input_msg.states.add()
        state.dtype = state_dtypes[self.do_preproc]
        state_obs = observation["state"]
        if isinstance(state_obs, torch.Tensor):
           state_obs=state_obs.detach().cpu().numpy()
        state.shape.extend(state_obs.shape)
        state.data = state_obs.tobytes()

        # 发送消息
        def send_proto_message(sock: socket.socket, msg, verbose: bool) -> bool:
            try:
                # 1. 序列化 Protobuf 消息
                serialized_data = msg.SerializeToString()
                data_len = len(serialized_data)

                # 2. 长度字段按大端打包（4 字节），与 wire.py 一致
                net_len_bytes = data_len.to_bytes(4, "big")

                assert len(net_len_bytes) == 4, f"长度字段应为4字节，实际{len(net_len_bytes)}字节"

                # 3. 先发送长度，再发送数据
                sock.sendall(net_len_bytes)  # 发送 4 字节长度
                sock.sendall(serialized_data)  # 发送 Protobuf 数据
                if verbose:
                    _print_io_verbose("SEND", True, data_len, msg.header, msg)
                elif live_io is not None:
                    live_io.complete_send(True)
                return True
            except Exception as e:
                print(f"发送失败：{str(e)}")
                if verbose:
                    _print_io_verbose("SEND", False, 0)
                elif live_io is not None:
                    live_io.complete_send(False)
                return False

        return send_proto_message(sock, input_msg, verbose)

    def receive(self, sent_obs=None, reset=None, verbose: bool = True, live_io: EvalLiveProgress | None = None):
        def recv_proto_message(sock, msg, verbose: bool):
            """接收protobuf消息(先接收长度，再接收数据)"""
            try:
                # 1. 接收 4 字节长度（网络序→大端）
                net_len_data = sock.recv(4)
                if len(net_len_data) != 4:
                    print("未收到完整长度（需4字节，实际收到{}字节）".format(len(net_len_data)))
                    if verbose:
                        _print_io_verbose("RECEIVE", False, len(net_len_data))
                    elif live_io is not None:
                        live_io.complete_recv(False)
                    return False

                # 大端解析 4 字节长度，与 wire.py 一致
                data_len = int.from_bytes(net_len_data, "big")

                # 接收数据
                serialized_data = b""
                while len(serialized_data) < data_len:
                    chunk = sock.recv(min(4096, data_len - len(serialized_data)))
                    if not chunk:
                        print("连接断开")
                        if verbose:
                            _print_io_verbose("RECEIVE", False, len(serialized_data))
                        elif live_io is not None:
                            live_io.complete_recv(False)
                        return False
                    serialized_data += chunk

                # 反序列化
                msg.ParseFromString(serialized_data)
                if verbose:
                    _print_io_verbose("RECEIVE", True, data_len, msg.header, msg)
                elif live_io is not None:
                    live_io.complete_recv(True)
                return True
            except Exception as e:
                print(f"接收失败：{str(e)}")
                if verbose:
                    _print_io_verbose("RECEIVE", False, 0)
                elif live_io is not None:
                    live_io.complete_recv(False)
                return False

        while True:
            if self.sock_fd is None:
                self.connect()
                if self.sock_fd is None:
                    time.sleep(0.5)
                    continue
                if sent_obs is not None and not self.send(sent_obs, reset, verbose=verbose, live_io=live_io):
                    self.disconnect()
                    time.sleep(0.5)
                    continue

            batch = msg_pb2.MultiModalInput()
            if not recv_proto_message(self.sock_fd, batch, verbose):
                print("连接异常，等待客户端重连...")
                self.disconnect()
                self.connect()
                if self.sock_fd is None:
                    time.sleep(0.5)
                    continue
                if sent_obs is not None and not self.send(sent_obs, reset, verbose=verbose, live_io=live_io):
                    self.disconnect()
                    time.sleep(0.5)
                    continue
                continue

            def parse_type(input):
                if input.dtype == msg_pb2.Tensor.STRING:
                    arr = input.data
                    return arr.decode("utf-8")
                elif input.dtype == msg_pb2.Tensor.UINT8:
                    arr = np.frombuffer(input.data, dtype=np.uint8)
                elif input.dtype == msg_pb2.Tensor.FLOAT64:
                    arr = np.frombuffer(input.data, dtype=np.float64)
                elif input.dtype == msg_pb2.Tensor.FLOAT32:
                    arr = np.frombuffer(input.data, dtype=np.float32)
                elif input.dtype == msg_pb2.Tensor.INT32:
                    arr = np.frombuffer(input.data, dtype=np.int32)
                elif input.dtype == msg_pb2.Tensor.FP16:
                    arr = np.frombuffer(input.data, dtype=np.float16)

                return arr.reshape(input.shape)

            # 解析图片张量
            imgs = []
            img_size = len(batch.images)
            for i in range(img_size):
                img = batch.images[i]
                imgs.append(parse_type(img))

            # 解析语言张量
            langs = []
            lang_size = len(batch.languages)
            for i in range(lang_size):
                lang = batch.languages[i]
                langs.append(parse_type(lang))

            # 解析状态张量
            states = []
            state_size = len(batch.states)
            for i in range(state_size):
                state = batch.states[i]
                states.append(parse_type(state))

            img_keys = [
                ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
            ]

            obs = {}
            if imgs:
                obs["images"] = dict(zip(img_keys[self.do_preproc], imgs))
            if states:
                obs["state"] = states[0]
            if langs:
                obs["prompt"] = langs[0]

            return obs

        return None

    def proc_recv(self, recv_data):
        return recv_data

    def proc_action(self, recv_data):
        if self.do_postproc:
            action_in = np.array(recv_data["prompt"], dtype=np.float16)
        else:
            action_in = np.array(recv_data["prompt"], dtype=np.float64)
        if self._is_pytorch_model:
            action_jax = torch.tensor(action_in)
        else:
            action_jax = jax.tree.map(lambda x: jnp.array(x), action_in)
        action_jax = action_jax.squeeze()[: self.chunk].cpu()
        return action_jax

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
                self.send(sent_obs, reset, verbose=True)
                recv_data = self.receive(sent_obs, reset, verbose=True)
                eval_live.complete_cycle_verbose(cycle, env_step, env_step_lim)
            else:
                eval_live.begin_infer_cycle(cycle, env_step, env_step_lim)
                eval_live.begin_send()
                self.send(sent_obs, reset, verbose=False, live_io=eval_live)
                eval_live.begin_recv()
                recv_data = self.receive(sent_obs, reset, verbose=False, live_io=eval_live)
            self._io_log_cycles += 1
            action_recv = self.proc_action(recv_data)
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
            self.send(obs)
            recv_data = self.receive()
            obs_old = obs
            obs = self.proc_recv(recv_data)
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
                    np.save("test/scp/py_act_raw.npy", np.array(action_raw))
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
