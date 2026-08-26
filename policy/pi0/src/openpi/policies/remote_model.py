"""Remote action source: TCP/protobuf bridge extracted from policies/policy.py.

Policy delegates all network I/O (connect/disconnect/send/receive/proc_action)
to RemoteModel and, in network mode, swaps the local ``action = model(obs)``
site for ``action = remote_model(obs)`` via ``RemoteModel.__call__``.

Wire format mirrors openpi/patch/wire.py: 4-byte big-endian length prefix +
serialized msg_pb2.MultiModalInput body. IO logging uses a single overwriting
line (carriage-return + clear-line) so the screen never floods — one
send/recv status line persists per episode, updating in place.
"""
import select
import socket
import time

import jax.numpy as jnp
import numpy as np
import torch

from openpi.patch import msg_pb2
from openpi.policies.eval_progress import EvalLiveProgress

_BLUE = "\033[34m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_GRAY = "\033[90m"
_YELLOW = "\033[93m"
_MAGENTA = "\033[35m"
_RESET = "\033[0m"

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


def _flush_progress_line() -> None:
    print("\r\033[K", end="", flush=True)


def _format_body_summary(batch) -> str:
    """Compact one-line summary of the message body."""
    parts: list[str] = []
    if batch.images:
        img = batch.images[0]
        parts.append(
            f"img:{len(batch.images)}({_dtype_name(img.dtype)},{_shape_str(img.shape)})"
        )
    if batch.languages:
        lang = batch.languages[0]
        parts.append(
            f"lang:{len(batch.languages)}({_dtype_name(lang.dtype)},{_shape_str(lang.shape)})"
        )
    if batch.states:
        state = batch.states[0]
        parts.append(
            f"state:{len(batch.states)}({_dtype_name(state.dtype)},{_shape_str(state.shape)})"
        )
    return " ".join(parts) if parts else ""


def _status_text(ok: bool) -> str:
    if ok:
        return f"{_GREEN}OK{_RESET}"
    return f"{_RED}FAIL{_RESET}"


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


def _print_io_compact(
    tag: str,
    ok: bool,
    payload_bytes: int,
    header=None,
    batch=None,
) -> None:
    """Print a single SEND/RECEIVE status line that overwrites the previous line.

    Uses ``\\r\\033[K`` (carriage-return + clear-line) so each call replaces
    whatever was on the current line — no scrolling, no flooding.
    """
    _flush_progress_line()
    status = _status_text(ok)
    parts: list[str] = [_io_tag_label(tag), status]
    if payload_bytes:
        parts.append(f"{_GRAY}{payload_bytes}B{_RESET}")
    if header is not None:
        parts.append(f"Seq:{header.seq}")
        if tag == "SEND" and header.reset:
            parts.append(f"{_YELLOW}reset{_RESET}")
    if batch is not None:
        summary = _format_body_summary(batch)
        if summary:
            parts.append(summary)
    print("  " + " | ".join(parts), end="", flush=True)


class RemoteModel:
    """TCP/protobuf bridge acting as a remote action source.

    Construct once with the runtime cfg values; call ``connect()`` to accept a
    single board client, then invoke the instance like a model:
    ``action = remote(obs)`` (see ``__call__``).
    """

    def __init__(
        self,
        *,
        port: int,
        do_preproc: bool,
        do_postproc: bool,
        visp: bool,
        chunk: int,
        debug: bool,
    ) -> None:
        self.port = port
        self.do_preproc = do_preproc
        self.do_postproc = do_postproc
        self.visp = visp
        self.chunk = chunk
        self.debug = debug

        self.listen_fd: socket.socket | None = None
        self.sock_fd: socket.socket | None = None
        self.seq = 0

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
        lang = observation["prompt"]
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
            state_obs = state_obs.detach().cpu().numpy()
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
                    _print_io_compact("SEND", True, data_len, msg.header, msg)
                elif live_io is not None:
                    live_io.complete_send(True, data_len)
                return True
            except Exception as e:
                print(f"发送失败：{str(e)}")
                if verbose:
                    _print_io_compact("SEND", False, 0)
                elif live_io is not None:
                    live_io.complete_send(False, 0)
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
                        _print_io_compact("RECEIVE", False, len(net_len_data))
                    elif live_io is not None:
                        live_io.complete_recv(False, 0)
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
                            _print_io_compact("RECEIVE", False, len(serialized_data))
                        elif live_io is not None:
                            live_io.complete_recv(False, 0)
                        return False
                    serialized_data += chunk

                # 反序列化
                msg.ParseFromString(serialized_data)
                if verbose:
                    _print_io_compact("RECEIVE", True, data_len, msg.header, msg)
                elif live_io is not None:
                    live_io.complete_recv(True, data_len)
                return True
            except Exception as e:
                print(f"接收失败：{str(e)}")
                if verbose:
                    _print_io_compact("RECEIVE", False, 0)
                elif live_io is not None:
                    live_io.complete_recv(False, 0)
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
        return action_in.squeeze()[: self.chunk]

    def __call__(
        self,
        observation: dict,
        *,
        reset: bool = False,
        verbose: bool = True,
        live_io: EvalLiveProgress | None = None,
    ):
        """Drop-in ``remote_model(obs)`` API for the network-backed policy.

        Sends ``observation``, waits for the reply, and returns the processed
        action — replacing ``action = model(obs)`` at the Policy's call site.

        When ``live_io`` is provided, send/recv status is rendered as a single
        overwriting line via ``EvalLiveProgress`` (no screen flooding). When
        ``live_io`` is None (e.g. OBS test mode), a compact single-line
        overwrite is printed directly.
        """
        use_verbose = verbose and live_io is None
        if live_io is not None:
            live_io.begin_send()
        self.send(observation, reset=reset, verbose=use_verbose, live_io=live_io)
        if live_io is not None:
            live_io.begin_recv()
        recv_data = self.receive(observation, reset=reset, verbose=use_verbose, live_io=live_io)
        return self.proc_action(recv_data)
