#!/usr/bin/env python3
"""Mock board server for the VLA demo (x86 dev loop, no board/HBM needed).

Mimics the C++ `oellm_vla_serving` REST + SSE contract so vla.html works
unchanged. Acts as a protobuf CLIENT to the RoboTwin Sim Bridge (like the
board does), runs a stub inference (zero action), and pushes the received
observation frames to browsers via SSE.

Topology:  vla.html  <-(SSE/REST)-  mock_board  <-(protobuf TCP)-  Sim Bridge  <-(SAPIEN)-  env
With --stub on the Bridge, the env is StubSimEnv and nothing needs SAPIEN.

REST contract (must match oellm_vla_serving):
  GET  /                          -> vla.html
  GET  /api/models                -> {status, models, current_model_id}
  POST /api/model/switch         body{model_id}
  POST /api/sim/connect          body{ip,port}
  POST /api/sim/disconnect
  POST /api/sim/control          body{cmd,task?,instruction?}
  GET  /api/sim/status
  GET  /api/sim/stream           -> text/event-stream (data: <json>\n\n)
SSE events:
  {"type":"frame","seq":N,"images":["data:image/bmp;base64,...",x3],
   "step":N,"perf":{"vla_ms":0.0},"fps":N}
  {"type":"event","event":"...","msg":"..."}
  {"type":"status","connected":bool,"running":bool,"step":N,"success":N,"total":N}
"""

import argparse
import base64
import json
import os
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))  # for openpi.patch
from openpi.patch import msg_pb2  # noqa: E402
from openpi.patch.wire import send_msg as _send_msg, recv_msg as _recv_msg  # noqa: E402


def build_action(pi0_step, state_dim):
    """Stub action: zeros [1, pi0_step, state_dim] FLOAT64."""
    m = msg_pb2.MultiModalInput()
    h = m.header
    h.seq = 1
    h.stamp.sec = int(time.time())
    h.stamp.nsec = 0
    h.frame_id = "camera_optical_frame"
    h.reset = False
    t = m.languages.add()
    t.dtype = msg_pb2.Tensor.FLOAT64
    t.shape.extend([1, int(pi0_step), int(state_dim)])
    t.data = np.zeros((1, pi0_step, state_dim), dtype=np.float64).tobytes()
    return m


def obs_to_images(msg):
    """Decode obs images -> list of RGB uint8 arrays (HWC, 224x224x3).

    Bridge sends CHW [3,H,W] (matches board online_receive). Mock board
    transposes back to HWC for BMP encoding. Also tolerates HWC input.
    """
    out = []
    for t in msg.images:
        if t.dtype != msg_pb2.Tensor.UINT8:
            out.append(np.zeros((224, 224, 3), dtype=np.uint8))
            continue
        shape = list(t.shape)
        arr = np.frombuffer(t.data, dtype=np.uint8)
        # CHW [3,H,W]?
        if len(shape) == 3 and shape[0] == 3:
            c, h, w = shape
            if arr.size == c * h * w:
                chw = arr.reshape(c, h, w)
                out.append(np.ascontiguousarray(np.transpose(chw, (1, 2, 0))))
                continue
        # HWC [H,W,3]?
        if len(shape) == 3:
            h, w, c = shape
            if arr.size == h * w * c:
                out.append(arr.reshape(h, w, c))
                continue
        out.append(np.zeros((224, 224, 3), dtype=np.uint8))
    return out


# ---------- BMP data-URL encoder (stdlib only) ----------
def rgb_to_bmp_data_url(rgb):
    h, w = rgb.shape[:2]
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    row_bytes = w * 3
    pad = (4 - row_bytes % 4) % 4
    stride = row_bytes + pad
    pix_size = stride * h
    buf = bytearray()
    buf += b"BM"
    buf += struct.pack("<I", 14 + 40 + pix_size)
    buf += struct.pack("<HH", 0, 0)
    buf += struct.pack("<I", 14 + 40)
    buf += struct.pack("<I", 40)
    buf += struct.pack("<i", w)
    buf += struct.pack("<i", h)
    buf += struct.pack("<HH", 1, 24)
    buf += struct.pack("<I", 0)
    buf += struct.pack("<I", pix_size)
    buf += struct.pack("<ii", 2835, 2835)
    buf += struct.pack("<II", 0, 0)
    bgr = rgb[:, :, ::-1]
    for y in range(h - 1, -1, -1):
        buf += bgr[y].tobytes()
        buf += b"\x00" * pad
    return "data:image/bmp;base64," + base64.b64encode(bytes(buf)).decode("ascii")


# ---------- state ----------
class BoardState:
    def __init__(self):
        self.lock = threading.RLock()
        self.model_id = "pi0_beat_block_hammer"
        self.sim_ip = None
        self.sim_port = None
        self.sim_sock = None
        self.connected = False
        self.running = False
        self.step = 0
        self.success = 0
        self.total = 0
        self.fps = 0.0
        self.worker_thread = None
        self.worker_stop = threading.Event()
        # SSE subscribers: list of handler objects with .send_frame(json_str)
        self.subs = []

    def status_dict(self):
        with self.lock:
            return {
                "connected": self.connected,
                "running": self.running,
                "step": self.step,
                "success": self.success,
                "total": self.total,
                "model_id": self.model_id,
                "fps": round(self.fps, 2),
            }

    def broadcast(self, obj):
        line = "data: " + json.dumps(obj) + "\n\n"
        dead = []
        with self.lock:
            subs = list(self.subs)
        for s in subs:
            try:
                s.write_sse(line)
            except Exception:
                dead.append(s)
        if dead:
            with self.lock:
                for s in dead:
                    if s in self.subs:
                        self.subs.remove(s)


BOARD = BoardState()


# ---------- data worker ----------
def _worker_loop(pi0_step, state_dim):
    st = BOARD
    while not st.worker_stop.is_set():
        if not st.connected or st.sim_sock is None:
            time.sleep(0.1)
            continue
        try:
            obs, _recv_size = _recv_msg(st.sim_sock)
        except Exception:
            st.connected = False
            st.broadcast({"type": "event", "event": "disconnected", "msg": "data socket closed"})
            break
        if obs is None:
            st.connected = False
            st.broadcast({"type": "event", "event": "disconnected", "msg": "bridge closed"})
            break
        reset = obs.header.reset
        if reset:
            st.broadcast({"type": "event", "event": "episode_start", "msg": "reset received"})
        t0 = time.time()
        imgs = obs_to_images(obs)
        data_urls = [rgb_to_bmp_data_url(im) for im in imgs]
        st.step = obs.header.seq
        # View-only frames: display without inference. The bridge sends these
        # during the action-chunk execution loop so the browser sees every
        # sim step, not just one per chunk.
        view_only = obs.header.view_only
        frame = {
            "type": "frame",
            "seq": obs.header.seq,
            "images": data_urls,
            "step": st.step,
        }
        if not view_only:
            frame["perf"] = {"vla_ms": 0.0}
            frame["fps"] = round(st.fps, 2)
        st.broadcast(frame)
        if view_only:
            continue  # skip stub inference + action send
        # stub inference + send action
        act = build_action(pi0_step, state_dim)
        try:
            _send_msg(st.sim_sock, act)
        except Exception:
            st.connected = False
            st.broadcast({"type": "event", "event": "disconnected", "msg": "action send failed"})
            break
        dt = time.time() - t0
        st.fps = 1.0 / dt if dt > 0 else 0.0
        st.broadcast({"type": "status", **st.status_dict()})
    print("[mock] worker exit", flush=True)


# ---------- HTTP ----------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n == 0:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        if self.path == "/" or self.path == "/vla.html":
            self._serve_html()
            return
        if self.path == "/api/models":
            self._json(
                200,
                {
                    "status": "ok",
                    "current_model_id": BOARD.model_id,
                    "models": [
                        {
                            "id": "pi0_beat_block_hammer",
                            "name": "Pi0 (mock)",
                            "kind": "vla",
                            "ready": True,
                            "loaded": True,
                        }
                    ],
                },
            )
            return
        if self.path == "/api/sim/status":
            self._json(200, {"status": "ok", **BOARD.status_dict()})
            return
        if self.path == "/api/sim/tasks":
            tasks = _fetch_bridge_tasks()
            self._json(200, {"status": "ok", "tasks": tasks})
        if self.path == "/api/sim/stream":
            self._serve_sse()
            return
        self._json(404, {"status": "error", "message": "not found"})

    def do_POST(self):
        b = BOARD
        if self.path == "/api/model/switch":
            body = self._body()
            with b.lock:
                b.model_id = body.get("model_id", b.model_id)
            self._json(200, {"status": "ok", "switch_cost_ms": 1})
            return
        if self.path == "/api/sim/connect":
            body = self._body()
            ip = body.get("ip")
            port = int(body.get("port", 30001))
            ok = _do_connect(ip, port)
            self._json(
                200 if ok else 400, {"status": "ok" if ok else "error", "message": "" if ok else "connect failed"}
            )
            return
        if self.path == "/api/sim/disconnect":
            _do_disconnect()
            self._json(200, {"status": "ok"})
            return
        if self.path == "/api/sim/control":
            body = self._body()
            _do_control(body)
            self._json(200, {"status": "ok"})
            return
        self._json(404, {"status": "error", "message": "not found"})

    def _serve_html(self):
        p = os.path.join(HERE, "vla.html")
        if not os.path.isfile(p):
            self._json(404, {"status": "error", "message": "vla.html missing"})
            return
        with open(p, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        with BOARD.lock:
            BOARD.subs.append(self)
        self.write_sse("data: " + json.dumps({"type": "event", "event": "connected", "msg": "sse attached"}) + "\n\n")
        try:
            while True:
                if self.wfile.closed:
                    break
                time.sleep(0.2)
        except Exception:
            pass
        finally:
            with BOARD.lock:
                if self in BOARD.subs:
                    BOARD.subs.remove(self)

    def write_sse(self, line):
        self.wfile.write(line.encode("utf-8"))
        self.wfile.flush()

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass


def _ctrl_url():
    ip = BOARD.sim_ip or "127.0.0.1"
    port = (BOARD.sim_port or 30001) + 1
    return f"http://{ip}:{port}"


def _fetch_bridge_json(path, timeout=3):
    try:
        import urllib.request

        with urllib.request.urlopen(_ctrl_url() + path, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _fetch_bridge_tasks():
    d = _fetch_bridge_json("/tasks")
    if d and isinstance(d.get("tasks"), list):
        return d["tasks"]
    return ["beat_block_hammer"]


def _status_poller(stop_ev):
    """Reflect the Bridge's success/total (source of truth) to browsers."""
    last = {"success": -1, "total": -1}
    while not stop_ev.is_set() and BOARD.connected:
        d = _fetch_bridge_json("/status")
        if d:
            with BOARD.lock:
                BOARD.success = int(d.get("success", BOARD.success))
                BOARD.total = int(d.get("total", BOARD.total))
                if d.get("running") is not None:
                    BOARD.running = bool(d.get("running"))
            if BOARD.success != last["success"] and last["success"] >= 0:
                BOARD.broadcast({"type": "event", "event": "episode_success", "msg": f"{BOARD.success}/{BOARD.total}"})
            elif BOARD.total != last["total"] and BOARD.success == last["success"] and last["total"] >= 0:
                BOARD.broadcast({"type": "event", "event": "episode_fail", "msg": f"{BOARD.success}/{BOARD.total}"})
            last = {"success": BOARD.success, "total": BOARD.total}
        stop_ev.wait(1.0)


def _do_connect(ip, port):
    b = BOARD
    if not ip:
        return False
    _do_disconnect()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((ip, port))
        s.settimeout(None)
    except Exception as e:
        print(f"[mock] connect {ip}:{port} failed: {e}", flush=True)
        return False
    with b.lock:
        b.sim_ip = ip
        b.sim_port = port
        b.sim_sock = s
        b.connected = True
        b.worker_stop.clear()
        b.worker_thread = threading.Thread(target=_worker_loop, args=(b._pi0_step, b._state_dim), daemon=True)
        b.worker_thread.start()
        b._poller_stop = threading.Event()
        threading.Thread(target=_status_poller, args=(b._poller_stop,), daemon=True).start()
    b.broadcast({"type": "event", "event": "connected", "msg": f"{ip}:{port}"})
    return True


def _do_disconnect():
    b = BOARD
    if getattr(b, "_poller_stop", None) is not None:
        b._poller_stop.set()
    with b.lock:
        s = b.sim_sock
        b.sim_sock = None
        b.connected = False
    b.worker_stop.set()
    if s is not None:
        try:
            s.close()
        except Exception:
            pass


def _do_control(body):
    b = BOARD
    cmd = body.get("cmd")
    task = body.get("task")
    instruction = body.get("instruction")
    # Forward to Bridge control port (data_port + 1).
    ctrl_port = (b.sim_port or 30001) + 1 if b.sim_port else 30002
    ip = b.sim_ip or "127.0.0.1"
    try:
        import urllib.request

        data = json.dumps({"task": task, "instruction": instruction}).encode()
        req = urllib.request.Request(
            f"http://{ip}:{ctrl_port}/episode/{cmd}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=3)
        if cmd in ("start", "reset"):
            with b.lock:
                b.running = True
        elif cmd == "stop":
            with b.lock:
                b.running = False
    except Exception as e:
        print(f"[mock] control {cmd} forward failed: {e}", flush=True)
        # Local-only fallback for stub loop without bridge control.
        with b.lock:
            if cmd in ("start", "reset"):
                b.running = True
            elif cmd == "stop":
                b.running = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--pi0_step", type=int, default=50)
    ap.add_argument("--state_dim", type=int, default=14)
    args = ap.parse_args()
    BOARD._pi0_step = args.pi0_step
    BOARD._state_dim = args.state_dim
    print(f"[mock] board server on http://{args.host}:{args.port}", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _do_disconnect()
        srv.shutdown()


if __name__ == "__main__":
    main()
