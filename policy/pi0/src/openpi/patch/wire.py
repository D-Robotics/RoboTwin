"""Unified protobuf wire helpers for sim_bridge socket protocol.

Frame format: 4-byte big-endian length prefix + serialized protobuf body.
Used by both sim_bridge (SAPIEN server side) and mock_board_server (x86 dev
loop). policy.py keeps its own send/receive with verbose/live_io logging.
"""
import socket

from openpi.patch import msg_pb2


def _recvall(sock: socket.socket, n: int):
    """Receive exactly n bytes from sock, or None on EOF."""
    chunks = []
    got = 0
    while got < n:
        b = sock.recv(min(n - got, 65536))
        if not b:
            return None
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def send_msg(sock: socket.socket, msg: msg_pb2.MultiModalInput):
    """Send a protobuf message: 4-byte big-endian length + serialized data."""
    data = msg.SerializeToString()
    sock.sendall(len(data).to_bytes(4, "big") + data)


def recv_msg(sock: socket.socket):
    """Receive a MultiModalInput: read 4-byte big-endian length, then body."""
    hdr = _recvall(sock, 4)
    if not hdr:
        return None
    n = int.from_bytes(hdr, "big")
    body = _recvall(sock, n)
    if body is None:
        return None
    msg = msg_pb2.MultiModalInput()
    msg.ParseFromString(body)
    return msg
