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
from openpi.models import msg_pb2

from safetensors.torch import load_file
import torch.nn as nn
import torch
import einops

BasePolicy: TypeAlias = _base_policy.BasePolicy

TEST, SKIP, OBS, PREPROC, SIGLIP, SIGLIP_PRJ, PALIGEMMA, PALIGEMMA_FULL, ACTION, ACTION_B, FULL= range(11)

# test self.stage
# SKIP: Robotwin raw procedure
# TEST: Send local input and receive SIGLIP result
# OBS: Send and receive raw obs to test mismatch
# PREPROC: Send and receive preprocessed obs to test mismatch
# SIGLIP: Send preprocessed obs and receive SIGLIP result
# SIGLIP_PRJ: Send preprocessed obs and receive SIGLIP result before Linear
# PALIGEMMA: Send preprocessed obs and receive kv_cache result
# PALIGEMMA_FULL: Send raw obs and receive kv_cache result


from scipy.signal import butter

class MultiChannelButterworth:
    def __init__(self, cutoff, fs, channels, order=2):
        self.b, self.a = butter(order, cutoff / (0.5 * fs), btype='low')
        self.order = order
        self.channels = channels
        self.x_hist = np.zeros((len(self.b), channels))
        self.y_hist = np.zeros((len(self.a), channels))

    def reset(self):
        self.x_hist = np.zeros((len(self.b), channels))
        self.y_hist = np.zeros((len(self.a), channels))
        
    def filter(self, x):

        x = np.asarray(x)
        assert x.shape == (self.channels,), f"Expected shape ({self.channels},), got {x.shape}"
        # Shift history 
        
        self.x_hist[1:] = self.x_hist[:-1]
        self.x_hist[0] = x

        self.y_hist[1:] = self.y_hist[:-1]

        # Compute output per channel
        y = (self.b[:, None] * self.x_hist).sum(axis=0) - \
            (self.a[1:, None] * self.y_hist[1:]).sum(axis=0)
        y /= self.a[0]
         
        self.y_hist[0] = y

        return y
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
        cfg = None
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
        self.stage = cfg['stage']
        self.port = cfg['port']
        self.use_raw = self.stage not in [OBS,PALIGEMMA_FULL,ACTION,FULL]
        
        # filter
        fs = 50       # 采样率 50Hz
        cutoff = 1    # 截止频率 5Hz
        channels = 14 if self.stage == FULL else 32 # 三通道数据（如加速度 X/Y/Z）
        self.filter = MultiChannelButterworth(cutoff, fs, channels)

    def filter(self,arr):
        filtered = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            filtered[i,:] = self.filter.filter(arr[i,:])
        return filtered

    def reset_filter(self):
        self.filter.reset()
    
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
            # 1. 创建 TCP Socket（对应 C++ 的 socket(AF_INET, SOCK_STREAM, 0)）
            try:
                # SOCK_STREAM 表示 TCP 协议，AF_INET 表示 IPv4
                self.listen_fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 设置端口复用（避免 TIME_WAIT 导致端口无法重启，对应 C++ 的 SO_REUSEADDR）
                self.listen_fd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except OSError as e:
                print(f"创建 Socket 失败：{str(e)}")
                return
            
            # 2. 绑定端口（对应 C++ 的 bind()，监听 8888 端口）
            server_addr = ("0.0.0.0", self.port)  # 0.0.0.0 等价于 C++ 的 INADDR_ANY（监听所有网卡）
            try:
                self.listen_fd.bind(server_addr)
            except OSError as e:
                print(f"绑定端口失败：{str(e)}")
                self.listen_fd.close()
                return

            # 3. 开始监听连接（对应 C++ 的 listen()，backlog=5）
            try:
                self.listen_fd.listen(5)  # backlog：等待队列最大长度
                print(f"服务器启动成功，等待客户端连接...（端口：{self.port}）")
            except OSError as e:
                print(f"监听失败：{str(e)}")
                self.listen_fd.close()
                return
        
            # 4. 接受客户端连接（对应 C++ 的 accept()，阻塞直到有连接）
            try:
                # client_addr：存储客户端地址信息，addr_len：地址长度
                self.sock_fd, client_addr = self.listen_fd.accept()
                print(f"客户端已连接：IP={client_addr[0]}, 端口={client_addr[1]}")
            except OSError as e:
                print(f"接受连接失败：{str(e)}")
                self.listen_fd.close()
                return
            
        listen()
        
    def send(self, observation:dict, reset=False):
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
                    img = jnp.moveaxis(img,3,1)
                    print('shape sent:',img.shape)
                    image.dtype = msg_pb2.Tensor.FP16
                image.shape.extend(img.shape)
                image.data = img.tobytes()
        else:
            # send kvcache
            img = observation["images"]
            image = input_msg.images.add()
            img = img.astype(jnp.float32)
            image.dtype = msg_pb2.Tensor.FLOAT32
            image.shape.extend(img.shape)
            image.data = img.tobytes()

        # 添加语言Tensor
        language = input_msg.languages.add()
        language.dtype = lang_dtypes[self.use_raw]
        language.shape.extend(observation["prompt"].shape)
        language.data = observation["prompt"].encode("utf-8") if not self.use_raw else observation["prompt"].astype(np.float32).tobytes()
               
        # 添加状态Tensor
        state = input_msg.states.add()
        state.dtype = state_dtypes[self.use_raw]
        print(observation["state"].dtype)
        state.shape.extend(observation["state"].shape)
        state.data = observation["state"].tobytes()

        # 发送消息
        def send_proto_message(sock: socket.socket, msg) -> bool:
            try:
                # 1. 序列化 Protobuf 消息
                serialized_data = msg.SerializeToString()
                data_len = len(serialized_data)
                print(f"待发送数据长度：{data_len}字节")

                # 2. 关键：长度字段按“大端字节序”打包（与 C++ 网络序一致）
                net_len = socket.htonl(data_len)  # 主机序→网络序（大端）
                net_len_bytes = struct.pack("<I", net_len)  # 大端打包为 4 字节
                # 验证长度字段是否为 4 字节（必须满足）
                assert len(net_len_bytes) == 4, f"长度字段应为4字节，实际{len(net_len_bytes)}字节"

                print(f"待发送的长度字段（十六进制）：{net_len_bytes.hex()}")
                # 3. 先发送长度，再发送数据
                sock.sendall(net_len_bytes)  # 发送 4 字节长度
                print(f"发送的长度字段（十六进制）：{net_len_bytes.hex()}")
                
                sock.sendall(serialized_data)  # 发送 Protobuf 数据
                print(f"发送成功：长度字段4字节 + 数据{data_len}字节")
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
            print(f"消息序列号 (seq): {header.seq}")
            print(f"时间戳: {header.stamp.sec} 秒 {header.stamp.nsec} 纳秒")
            time_sec = header.stamp.sec + header.stamp.nsec / 1e9
            print(f"（等价于 {time_sec} 秒）")
            print(f"坐标系 ID (frame_id): {header.frame_id}")
            print("===========================")    
        
        def recv_proto_message(sock, msg):
            """接收protobuf消息(先接收长度，再接收数据)"""
            try:
                # 1. 接收 4 字节长度（网络序→大端）
                net_len_data = sock.recv(4)
                if len(net_len_data) != 4:
                    print("未收到完整长度（需4字节，实际收到{}字节）".format(len(net_len_data)))
                    return False

                # 关键：用 ">I"（大端）解析 4 字节无符号整数（与 C++ 的 htonl 对应）
                net_len = struct.unpack("<I", net_len_data)[0]
                # 网络序转主机序（若系统是小端，此步必须；大端系统可省略，但建议保留兼容性）
                data_len = socket.ntohl(net_len)

                print(f"解析到数据长度：{data_len}字节（等待接收）")  # 加日志验证长度是否合理

                # 接收数据
                serialized_data = b''
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
                    return arr.decode('utf-8')
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
            imgs= []
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
            langs= []
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
            states= []
            state_size = len(batch.states)
            print(f"接收到 {state_size} 个状态张量：")
            for i in range(state_size):
                state = batch.states[i]
                print(f"  语言{i}：类型={state.dtype}，维度=", end="")
                for dim in state.shape:
                    print(f"{dim} ", end="")
                print()
                states.append(parse_type(state))

            print("解析完成\n")

            img_keys = [['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                        ['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']]

            obs = {}
            if imgs:
                obs["images"]=dict(zip(img_keys[self.use_raw],imgs))
            if states:
                obs["state"]=states[0]
            if langs:
                obs["prompt"]=langs[0]
            
            return obs
        
        return None
    
    def proc_recv(self,recv_data):
        return recv_data

    def proc_siglip(self,recv_data):
        siglip_in = recv_data["images"]
        siglip_jax = jax.tree.map(
            lambda x: jnp.array(x, dtype=jnp.bfloat16),
            siglip_in
        )
        return siglip_jax

    def proc_action(self,recv_data):
        if self.stage == FULL:
            paligemma_in = np.array(recv_data["prompt"],dtype=np.float64)
        else:
            paligemma_in = np.array(recv_data["prompt"],dtype=np.float16)
        if self._is_pytorch_model:
            paligemma_jax = torch.tensor(paligemma_in)
        else:
            paligemma_jax = jax.tree.map(
                lambda x: jnp.array(x),
                paligemma_in
            )
        return paligemma_jax
    
    def proc_paligemma(self,recv_data):
        paligemma_in = np.array(recv_data["prompt"],dtype=np.float32)
        paligemma_out = (paligemma_in[:18],paligemma_in[18:])

        paligemma_jax = jax.tree.map(
            lambda x: jnp.array(x, dtype=jnp.bfloat16),
            paligemma_out
        )
        return paligemma_jax
    
    @override
    def infer(self, obs: dict,reset=False, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # PATCH
        def dict_equal(d1, d2, atol=1e-6):
            if not (isinstance(d1, dict) and isinstance(d2, dict)):
                return False
            if d1.keys() != d2.keys():
                return False
            for key in d1:
                v1, v2 = d1[key], d2[key]
                if isinstance(v1, dict) and isinstance(v2, dict):
                    if not dict_equal(v1, v2, atol=atol):
                        return False
                elif isinstance(v1, np.ndarray):
                    v2 = np.array(v2)
                    v2 = np.squeeze(v2)
                    if v1.shape != v2.shape:
                        print(v1.shape, v2.shape)
                        return False
                    if not np.allclose(v1, v2, atol=atol):
                        with open('1.txt','w') as f:
                            for i in range(v1.shape[0]):
                                if not np.allclose(v1[i], v2[i], atol=atol):
                                    f.write(str(v1[i])+'\n')
                                    f.write(str(v2[i])+'\n')
                                    print(i)
                                    break

                        return False
                elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
                    if len(v1) != len(v2):
                        return False
                    for elem1, elem2 in zip(v1, v2):
                        if not dict_equal(elem1, elem2, atol=atol):
                            return False
                else:
                    if v1 != v2:
                        return False
                return True

        if self.stage != SKIP:
            self.connect()

        if self._model is None:
            self.send(obs,reset)
            recv_data = self.receive()
            action_result = self.proc_action(recv_data)
            outputs = {"actions":action_result.squeeze()}
            np.save("test/cpp_act.npy",np.array(action_result))
            return outputs
            
        siglip_result = None
        kvcache_result = None
        action_result = None
        if self.stage == OBS:
            self.send(obs)
            recv_data = self.receive()
            obs_old = obs   
            obs = self.proc_recv(recv_data)
              
            assert dict_equal(obs,obs_old) ,"recv mismatch!"
        # PATCH END

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        # Make a batch and convert to jax.Array.
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

        if self.stage in [PREPROC, SIGLIP, SIGLIP_PRJ, PALIGEMMA, TEST]:
            obs = _model.Observation.from_dict(inputs)
            obs = _preprocessing.preprocess_observation_pytorch(obs ,train=False)
            obs = {
                'images':obs.images,
                'state':obs.state,
                'prompt':obs.tokenized_prompt
            }
            if self.stage == TEST:
                test_result_bchw= np.fromfile("/mnt/data/yanjie.shen/RoboTwin/policy/pi0/test/input.bin", dtype=np.float16).reshape(1,3,224,224)
                test_result = jnp.transpose(jnp.array(test_result_bchw), (0, 2, 3, 1))
                print('shape_in',test_result.shape)
                print('vector_in',test_result)
                obs["images"]['base_0_rgb'] = test_result
                obs["images"]['left_wrist_0_rgb'] = test_result
                obs["images"]['right_wrist_0_rgb'] = test_result
    
            self.send(obs)
            print('wait receive')
            recv_data = self.receive()
        
        if self.stage in [PALIGEMMA_FULL,ACTION,FULL]:
            self.send(obs,reset)
  
            obs = _model.Observation.from_dict(inputs)
            obs =  _preprocessing.preprocess_observation_pytorch(obs ,train=False)
            obs = {
                'images':obs.images,
                'state':obs.state,
                'prompt':obs.tokenized_prompt
            }   

            print('wait receive')           
            recv_data = self.receive()

        if self.stage == PREPROC:
            obs_old = obs
            obs = self.proc_recv(recv_data)
            assert dict_equal(obs,obs_old) ,"recv mismatch!"
        elif self.stage == SIGLIP or self.stage == SIGLIP_PRJ or self.stage == TEST:
            siglip_result = self.proc_siglip(recv_data)
        elif self.stage == PALIGEMMA or self.stage == PALIGEMMA_FULL:
            kvcache_result = self.proc_paligemma(recv_data)
        elif self.stage == ACTION or self.stage == FULL:
            action_result = self.proc_action(recv_data)

        kvcache_result, actions = self._sample_actions(sample_rng_or_pytorch_device, _model.Observation.from_dict(inputs), **self._sample_kwargs)

        if self.stage == ACTION_B:
            obs = _model.Observation.from_dict(inputs)
            obs =  _preprocessing.preprocess_observation_pytorch(obs ,train=False)
            obs = {
                'images':kvcache_result, # actually kv_cache
                'state':obs.state.cpu().numpy().astype(np.float32),
                'prompt':obs.tokenized_prompt.cpu().numpy()
            }
            self.send(obs)
            recv_data = self.receive()
            action_result = self.proc_action(recv_data)
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        self.count +=1

        if self.stage == ACTION or self.stage == ACTION_B:
           # print('raw action',outputs["actions"])
          #  print('cpp action',action_result)
            np.save("test/py_act.npy",np.array(outputs["actions"].detach().cpu().numpy()))
            np.save("test/cpp_act.npy",np.array(action_result))
          #  reset_filter()
          #  action_result = filter(action_result.squeeze()).unsqueeze(0)
            outputs["actions"] = action_result

        # Unbatch and convert to np.ndarray.
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if self.stage == FULL:
            np.save("test/py_act.npy",np.array(outputs["actions"]))
            np.save("test/cpp_act.npy",np.array(action_result))
          #  if (reset):
          #      reset_filter()
          #  action_result = filter(action_result.squeeze())
                
            outputs["actions"] = action_result.squeeze()
            
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
