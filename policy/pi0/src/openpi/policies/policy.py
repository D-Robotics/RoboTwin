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
import time

from safetensors.torch import load_file
import torch.nn as nn
import torch
import einops
import os

from openpi.patch import msg_pb2
from openpi.patch.filters import NoFilter, FIR, ZeroPhaseFTR, MultiChannelButterworth
from openpi.patch.utils import dict_equal

BasePolicy: TypeAlias = _base_policy.BasePolicy

OBS, SKIP, ACTION, FULL = range(4)
DBG = False

# SKIP: Robotwin raw procedure
# OBS: Send and receive obs to test consistency
# ACTION: Send obs and receive action_expert result(w/o postproc)
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

        self.listen_fd = None
        self.sock_fd = None
        self.seq = 0

        self.count = 0

        # config
        self.stage = cfg["stage"]
        self.port = cfg["port"]
        self.do_preproc = cfg["do_preproc"]
        self.use_raw = self.do_preproc and self.stage != FULL
        self.visp = cfg["visp"]
        self.chunk = cfg["chunk"]
        self.debug = cfg["debug"]
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

        self.filter = filter_class(**common_params)
        self.filter_py = filter_class(**common_params)

    def filt(self, arr):
        return self.filter.filter(arr)

    def filt_py(self, arr):
        return self.filter_py.filter(arr)

    def reset_filter(self):
        self.filter.reset()
        self.filter_py.reset()

    def connect(self):
        def is_connected(sock: socket.socket) -> bool:
            if sock is None:
                return False
            try:
                sock.getpeername()  # 如果未连接，会抛异常
                return True
            except socket.error:
                return False

        if is_connected(self.sock_fd):
            return

        def listen():
            # 1. 创建 TCP Socket
            try:
                # SOCK_STREAM 表示 TCP 协议，AF_INET 表示 IPv4
                self.listen_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 设置端口复用（避免 TIME_WAIT 导致端口无法重启，对应 C++ 的 SO_REUSEADDR）
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
                self.listen_fd.close()
                return

            # 3. 开始监听连接
            try:
                self.listen_fd.listen(5)  # backlog：等待队列最大长度
                print(f"服务器启动成功，等待客户端连接...（端口：{self.port}）")
            except OSError as e:
                print(f"监听失败：{str(e)}")
                self.listen_fd.close()
                return

            # 4. 接受客户端连接
            try:
                self.sock_fd, client_addr = self.listen_fd.accept()
                print(f"客户端已连接：IP={client_addr[0]}, 端口={client_addr[1]}")
            except OSError as e:
                print(f"接受连接失败：{str(e)}")
                self.listen_fd.close()
                return

        listen()

    def send(self, observation: dict, reset=False):
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
                image.dtype = img_dtypes[self.use_raw]
                if self.use_raw:
                    img = img.astype(jnp.float16)
                    img = jnp.moveaxis(img, 3, 1)
                    print("shape sent:", img.shape)
                    image.dtype = msg_pb2.Tensor.FP16
                image.shape.extend(img.shape)
                image.data = img.tobytes()
        else:
            # send kvcache
            img = observation["images"]
            image = input_msg.images.add()
            if self.visp:
                img = img.astype(jnp.float16)
                image.dtype = msg_pb2.Tensor.FP16
                print("sent fp16")
            else:
                img = img.astype(jnp.float32)
                image.dtype = msg_pb2.Tensor.FLOAT32
                print("sent fp32")
            image.shape.extend(img.shape)
            image.data = img.tobytes()

        # 添加语言Tensor
        language = input_msg.languages.add()
        language.dtype = lang_dtypes[self.use_raw]
        language.shape.extend(observation["prompt"].shape)
        language.data = (
            observation["prompt"].encode("utf-8")
            if not self.use_raw
            else observation["prompt"].astype(np.float32).tobytes()
        )

        # 添加状态Tensor
        state = input_msg.states.add()
        state.dtype = state_dtypes[self.use_raw]
        state.shape.extend(observation["state"].shape)
        state.data = observation["state"].tobytes()

        # 发送消息
        def send_proto_message(sock: socket.socket, msg) -> bool:
            try:
                # 1. 序列化 Protobuf 消息
                serialized_data = msg.SerializeToString()
                data_len = len(serialized_data)

                # 2. 关键：长度字段按大端字节序打包
                net_len = socket.htonl(data_len)  # 主机序→网络序（大端）
                net_len_bytes = struct.pack("<I", net_len)  # 大端打包为 4 字节

                assert len(net_len_bytes) == 4, f"长度字段应为4字节，实际{len(net_len_bytes)}字节"

                # 3. 先发送长度，再发送数据
                sock.sendall(net_len_bytes)  # 发送 4 字节长度
                sock.sendall(serialized_data)  # 发送 Protobuf 数据
                print(f"发送成功，长度：{data_len}字节\n")
                return True
            except Exception as e:
                print(f"发送失败：{str(e)}")
                return False

        send_proto_message(sock, input_msg)

    def receive(self):
        batch = msg_pb2.MultiModalInput()
        sock = self.sock_fd

        def parse_header(header):
            """解析并打印Header信息"""
            print("===== 解析 Header 信息 =====")
            print(f"序列号 : {header.seq}")
            print(f"时间戳: {header.stamp.sec}.{header.stamp.nsec}")
            time_sec = header.stamp.sec + header.stamp.nsec / 1e9
            print("====== 解析 Body 信息 ======")

        def recv_proto_message(sock, msg):
            """接收protobuf消息(先接收长度，再接收数据)"""
            try:
                # 1. 接收 4 字节长度（网络序→大端）
                net_len_data = sock.recv(4)
                if len(net_len_data) != 4:
                    print("未收到完整长度（需4字节，实际收到{}字节）".format(len(net_len_data)))
                    return False

                # 关键：用 ">I"（大端）解析 4 字节无符号整数
                net_len = struct.unpack("<I", net_len_data)[0]
                # 网络序转主机序（小端系统必须，大系端统可省略）
                data_len = socket.ntohl(net_len)

                # 接收数据
                serialized_data = b""
                while len(serialized_data) < data_len:
                    chunk = sock.recv(min(4096, data_len - len(serialized_data)))
                    if not chunk:
                        print("连接断开")
                        return False
                    serialized_data += chunk

                # 反序列化
                msg.ParseFromString(serialized_data)
                print(f"接收成功，长度：{data_len}字节")
                return True
            except Exception as e:
                print(f"接收失败：{str(e)}")
                return False

        if recv_proto_message(sock, batch):
            # 解析Header
            parse_header(batch.header)

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
            print(f"接收到 {img_size} 个图像张量：")
            for i in range(img_size):
                img = batch.images[i]
                print(f"  语言{i}：类型={img.dtype}，维度=", end="")
                for dim in img.shape:
                    print(f"{dim} ", end="")
                print()
                imgs.append(parse_type(img))

            # 解析语言张量
            langs = []
            lang_size = len(batch.languages)
            print(f"接收到 {lang_size} 个嵌入张量：")
            for i in range(lang_size):
                lang = batch.languages[i]
                print(f"  语言{i}：类型={lang.dtype}，维度=", end="")
                for dim in lang.shape:
                    print(f"{dim} ", end="")
                print()
                langs.append(parse_type(lang))

            # 解析状态张量
            states = []
            state_size = len(batch.states)
            print(f"接收到 {state_size} 个状态张量：")
            for i in range(state_size):
                state = batch.states[i]
                print(f"  语言{i}：类型={state.dtype}，维度=", end="")
                for dim in state.shape:
                    print(f"{dim} ", end="")
                print()
                states.append(parse_type(state))

            print("============================")

            img_keys = [
                ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
            ]

            obs = {}
            if imgs:
                obs["images"] = dict(zip(img_keys[self.use_raw], imgs))
            if states:
                obs["state"] = states[0]
            if langs:
                obs["prompt"] = langs[0]

            return obs

        return None

    def proc_recv(self, recv_data):
        return recv_data

    def proc_action(self, recv_data):
        if self.stage == FULL:
            action_in = np.array(recv_data["prompt"], dtype=np.float64)
        else:
            action_in = np.array(recv_data["prompt"], dtype=np.float16)
        if self._is_pytorch_model:
            action_jax = torch.tensor(action_in)
        else:
            action_jax = jax.tree.map(lambda x: jnp.array(x), action_in)
        action_jax = action_jax.squeeze()[: self.chunk]
        return action_jax

    @override
    def infer(self, obs: dict, reset=False, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        if self.stage != SKIP:
            self.connect()

        if reset:
            self.reset_filter()

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        action_result = None
        if self._model is None:
            self.send(obs, reset)
            recv_data = self.receive()
            action_result = self.proc_action(recv_data).squeeze()
            if self.stage == ACTION:
                inputs = self._input_transform(inputs)
                outputs = {"actions": action_result, "state": inputs["state"]}
                outputs = self._output_transform(outputs)
            else:
                outputs = {"actions": action_result}
            outputs["actions"] = self.filt(outputs["actions"])
            return outputs

        if self.stage == OBS:
            self.send(obs)
            recv_data = self.receive()
            obs_old = obs
            obs = self.proc_recv(recv_data)

            assert dict_equal(obs, obs_old), "recv mismatch!"

        # Make a batch and convert to jax.Array.
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if self.do_preproc and self.stage != SKIP:
            obs = _model.Observation.from_dict(inputs)
            obs = _preprocessing.preprocess_obsservation_pytorch(obs, train=False)
            obs = {"images": obs.images, "state": obs.state, "prompt": obs.tokenized_prompt}

            self.send(obs)
            print("wait receive")
            recv_data = self.receive()

        if not self.do_preproc and self.stage != SKIP:
            self.send(obs, reset)

            obs = _model.Observation.from_dict(inputs)
            obs = _preprocessing.preprocess_observation_pytorch(obs, train=False)
            obs = {"images": obs.images, "state": obs.state, "prompt": obs.tokenized_prompt}

            print("wait receive")
            recv_data = self.receive()

        if self.stage == OBS:
            obs_old = obs
            obs = self.proc_recv(recv_data)
            assert dict_equal(obs, obs_old), "recv mismatch!"
        elif self.stage in (ACTION, FULL):
            action_result = self.proc_action(recv_data)

        _, action_result = self._sample_actions(
            sample_rng_or_pytorch_device, _model.Observation.from_dict(inputs), **self._sample_kwargs
        )

        outputs = {
            "state": inputs["state"],
            "actions": action_result,
        }
        self.count += 1

        if self.stage == ACTION:
            if self.debug:
                print("raw action", outputs["actions"])
                print("cpp action", action_result)
                np.save("test/py_act.npy", np.array(outputs["actions"].detach().cpu().numpy()))
                np.save("test/cpp_act.npy", np.array(action_result))

            outputs["actions"] = action_result

        # Unbatch and convert to np.ndarray.
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        act_raw = outputs["actions"].squeeze()

        if self.stage == FULL:
            outputs["actions"] = action_result.squeeze()
            if self.debug:
                np.save("test/scp/py_act_raw.npy", np.array(act_raw))
                np.save("test/py_act.npy", np.array(act_raw))
                np.save("test/cpp_act.npy", np.array(action_result))

        outputs["actions"] = self.filt(outputs["actions"].squeeze())

        if self.debug:
            filted_py = self.filt_py(act_raw)
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
