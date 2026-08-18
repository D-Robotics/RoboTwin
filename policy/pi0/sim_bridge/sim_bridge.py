#!/usr/bin/env python3
"""RoboTwin Sim Bridge for the VLA visualization demo.

Two servers in one process:
  - TCP data server  (default :30001, protobuf MultiModalInput ping-pong)
  - HTTP control     (default :30002, /tasks /episode/* /status /config)

Role: the SERVER side of the protocol used by
llm_engine/demo/vla_sdk_demo/common/src/vla_demo_network.h, where the board
side is the CLIENT (connect(server_ip, server_port)).

Data loop per connected board client:
  1. Wait until running (control /episode/start).
  2. On pending reset: env.reset(task, seed, instruction); next obs carries
     header.reset=True so the board resets its observation window too.
  3. obs = env.get_obs(); build & send MultiModalInput(images[3], lang, state).
  4. recv action MultiModalInput(languages[0] tensor [N, chunk, dim]).
  5. Execute pi0_step sub-actions via env.take_action; check eval_success.
  6. Goto 3.

Run from the RoboTwin repo root (see run_bridge.sh). --stub uses StubSimEnv so
the loop runs with no SAPIEN/RoboTwin deps (x86 demo loop).
"""

import argparse
import json
import os
import socket
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # for env_wrapper
sys.path.insert(0, os.path.join(_HERE, "..", "src"))  # for openpi.patch
from openpi.patch import msg_pb2  # noqa: E402
from openpi.patch.wire import send_msg as _send_msg, recv_msg as _recv_msg  # noqa: E402
from env_wrapper import SimEnv, StubSimEnv  # noqa: E402


def _now_stamp():
    t = time.time()
    return int(t), int((t - int(t)) * 1e9)


def build_obs(rgb_list, state, instruction, reset, seq, view_only=False):
    msg = msg_pb2.MultiModalInput()
    h = msg.header
    h.seq = seq
    sec, nsec = _now_stamp()
    h.stamp.sec = sec
    h.stamp.nsec = nsec
    h.frame_id = "camera_optical_frame"
    h.reset = reset
    h.view_only = view_only
    for rgb in rgb_list:
        t = msg.images.add()
        t.dtype = msg_pb2.Tensor.UINT8
        # Board online_receive parses images as CHW (layout=kChw,
        # height=shape[-2], width=shape[-1]); openpi sends [C,H,W]. Match
        # that: transpose HWC->[3,H,W] and shape [C,H,W].
        chw = np.ascontiguousarray(
            np.transpose(rgb, (2, 0, 1)), dtype=np.uint8)
        t.shape.extend([3, int(rgb.shape[0]), int(rgb.shape[1])])
        t.data = chw.tobytes()
    lang = msg.languages.add()
    lang.dtype = msg_pb2.Tensor.STRING
    lang.data = (instruction or "").encode("utf-8")
    st = msg.states.add()
    st.dtype = msg_pb2.Tensor.FLOAT64
    st.shape.extend([int(len(state))])
    st.data = np.asarray(state, dtype=np.float64).tobytes()
    return msg


def parse_action(msg):
    """Return np.ndarray shaped [N, chunk, dim] from the action message."""
    if not msg.languages:
        return None
    t = msg.languages[0]
    if t.dtype == msg_pb2.Tensor.FLOAT64:
        dt = np.float64
    elif t.dtype == msg_pb2.Tensor.FP16:
        dt = np.float16
    elif t.dtype == msg_pb2.Tensor.FLOAT32:
        dt = np.float32
    else:
        return None
    arr = np.frombuffer(t.data, dtype=dt).copy()
    shape = [int(s) for s in t.shape]
    if arr.size != int(np.prod(shape)) if shape else arr.size:
        # tolerate size mismatch: reshape to last dim only
        shape = [arr.size // (shape[-1] if shape else 1), shape[-1] if shape else 1]
    return arr.reshape(shape)


class BridgeState:
    def __init__(self):
        self.lock = threading.RLock()
        # Serializes ALL SimEnv method calls (reset/get_obs/take_action)
        # across concurrent data-handler threads. The board may hold two
        # data connections open at once; without this, two threads can enter
        # env.reset() together and race inside warp's first-time kernel
        # compilation (a fresh docker container has a cold kernel cache),
        # which corrupts the module build and fails the reset with errors
        # like "Referencing undefined symbol" / "CUDA kernel build failed".
        self.env_lock = threading.Lock()
        self.task = "beat_block_hammer"
        self.instruction = ""
        self.running = False  # episode loop active
        self.paused = False
        self.pending_reset = True
        self.episode_active = False
        self.step = 0
        self.success = 0
        self.total = 0
        self.episode_id = 0  # increments on each reset; polled by the board
        self.last_result = ""  # "success"/"fail" of the just-finished episode
        self.last_steps = 0  # step count when the last episode ended
        self.last_frame_time = 0.0
        self.fps = 0.0
        self.client_addr = None
        self.stop_flag = False  # process shutdown
        self.view_stride = 1  # send view-only frame every N action steps
        self.data_srv = None  # set by main(); shutdown() to exit serve_forever

    def status_dict(self):
        with self.lock:
            return {
                "connected": self.client_addr is not None,
                "running": self.running and not self.paused,
                "paused": self.paused,
                "task": self.task,
                "instruction": self.instruction,
                "step": self.step,
                "success": self.success,
                "total": self.total,
                "fps": round(self.fps, 2),
                "episode_id": self.episode_id,
                "last_result": self.last_result,
                "last_steps": self.last_steps,
            }


class DataHandler(socketserver.BaseRequestHandler):
    bridge: "BridgeState" = None  # set as class attr by factory
    env = None

    def handle(self):
        b = self.bridge
        with b.lock:
            b.client_addr = f"{self.client_address[0]}:{self.client_address[1]}"
        print(f"[bridge] board connected from {b.client_addr}", flush=True)
        seq = 0
        try:
            while not b.stop_flag:
                if not b.running:
                    time.sleep(0.1)
                    continue
                performed_reset = False
                if b.pending_reset or not b.episode_active:
                    # One reset at a time (b.env_lock); re-check the flags
                    # after taking the lock — the other data handler may have
                    # completed the reset while we were waiting for it.
                    with b.env_lock:
                        if b.pending_reset or not b.episode_active:
                            # Stop once we've run test_num episodes (from
                            # config.yaml), mirroring eval_policy.py's
                            # `while succ_seed < test_num`. Without this the
                            # bridge runs past the last eval_data file into a
                            # FileNotFoundError on every reset.
                            test_num = getattr(self.env, "test_num", 0)
                            ep_num = getattr(self.env, "ep_num", 0)
                            if test_num > 0 and ep_num >= test_num:
                                print(f"[bridge] all {test_num} episodes done "
                                      f"(success {b.success}/{b.total}), "
                                      f"shutting down", flush=True)
                                b.stop_flag = True
                                break
                            try:
                                self.env.reset(task=b.task, seed=0,
                                               instruction=b.instruction)
                                b.episode_id += 1
                                b.episode_active = True
                                b.pending_reset = False
                                b.step = 0
                                performed_reset = True
                            except Exception as e:
                                # Exit immediately instead of retry-looping
                                # the same failed reset forever. The common
                                # cause is a missing
                                # eval_data/<task>/<sample>/<N>_pos.pkl
                                # (episode index past test_num, or data not
                                # generated yet). Retrying can never succeed —
                                # regenerate the eval_data with eval_policy.py
                                # / play_once first.
                                print(f"[bridge] env.reset failed: {e}",
                                      flush=True)
                                print("[bridge] exiting due to reset failure "
                                      "(regenerate eval_data with "
                                      "eval_policy.py if data files are "
                                      "missing)", flush=True)
                                b.stop_flag = True
                                break
                    reset = performed_reset
                else:
                    reset = False
                try:
                    with b.env_lock:
                        rgb, state, instr = self.env.get_obs()
                except Exception as e:
                    print(f"[bridge] get_obs failed: {e}", flush=True)
                    time.sleep(0.2)
                    continue
                seq += 1
                obs_msg = build_obs(rgb, state, instr, reset, seq)
                try:
                    _send_msg(self.request, obs_msg)
                except Exception:
                    break
                t0 = time.time()
                act_msg = _recv_msg(self.request)
                if act_msg is None:
                    print("[bridge] board closed data socket", flush=True)
                    break
                actions = parse_action(act_msg)
                if actions is not None and actions.size:
                    # actions shape [N, chunk, dim]; take first batch's chunks.
                    if actions.ndim == 3:
                        chunks = actions[0]
                    elif actions.ndim == 2:
                        chunks = actions
                    else:
                        chunks = actions.reshape(1, -1)
                    for idx, a in enumerate(chunks):
                        # Let the entire chunk finish before a pause takes
                        # effect — breaking mid-chunk leaves the sim in a
                        # partial state and discards the rest of the action
                        # the board already paid inference cost for. The
                        # while-loop top checks b.running after the chunk.
                        try:
                            with b.env_lock:
                                self.env.take_action(a)
                        except Exception as e:
                            print(f"[bridge] take_action failed: {e}",
                                  flush=True)
                            break
                        b.step += 1
                        # Send a view-only frame so the browser sees every
                        # sim step, not just one per inference chunk. Skip
                        # when paused (b.running=False) so the board's TCP
                        # buffer doesn't fill up while its worker is stopped.
                        if b.running and b.view_stride > 0 and \
                                (idx + 1) % b.view_stride == 0:
                            try:
                                with b.env_lock:
                                    rgb_v, state_v, _ = self.env.get_obs()
                                seq += 1
                                _send_msg(self.request,
                                          build_obs(rgb_v, state_v, instr,
                                                    False, seq,
                                                    view_only=True))
                            except Exception:
                                pass  # view-only send failure is non-fatal
                        if self.env.eval_success():
                            break
                        if (hasattr(self.env, "step_lim")
                                and self.env.step_lim
                                and self.env.take_action_cnt
                                >= self.env.step_lim):
                            break
                    if self.env.eval_success():
                        b.success += 1
                        b.total += 1
                        b.last_result = "success"
                        b.last_steps = b.step
                        b.episode_active = False
                        b.pending_reset = True  # auto-next episode
                        print(f"[bridge] episode success "
                              f"({b.success}/{b.total})", flush=True)
                    elif (hasattr(self.env, "step_lim")
                          and self.env.step_lim
                          and self.env.take_action_cnt
                          >= self.env.step_lim):
                        b.total += 1
                        b.last_result = "fail"
                        b.last_steps = b.step
                        b.episode_active = False
                        b.pending_reset = True
                        print(f"[bridge] episode fail "
                              f"({b.success}/{b.total})", flush=True)
                b.last_frame_time = time.time() - t0
                b.fps = 1.0 / b.last_frame_time if b.last_frame_time > 0 else 0.0
        finally:
            with b.lock:
                b.running = False
                b.paused = False
                # Treat disconnect as a pause, not an episode end: keep
                # episode_active / pending_reset as-is so reconnect+start
                # resumes the current episode instead of starting a new one.
                b.client_addr = None
            # If the loop exited via stop_flag (episodes exhausted or reset
            # failure), unblock serve_forever() in the main thread so the
            # process actually exits instead of idling on the listen socket.
            if b.stop_flag and b.data_srv is not None:
                try:
                    b.data_srv.shutdown()
                except Exception:
                    pass
            print("[bridge] board data handler exit", flush=True)


class _DataServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_data_handler(bridge, env):
    DataHandler.bridge = bridge
    DataHandler.env = env
    return DataHandler


class ControlHandler(BaseHTTPRequestHandler):
    bridge: "BridgeState" = None
    env = None

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **k):
        pass

    def do_GET(self):
        b = self.bridge
        if self.path == "/tasks":
            try:
                tasks = self.env.list_tasks()
            except Exception:
                tasks = [b.task]
            self._send(200, {"status": "ok", "tasks": tasks})
        elif self.path == "/status":
            self._send(200, {"status": "ok", **b.status_dict()})
        elif self.path == "/config":
            self._send(200, {"status": "ok",
                             "config": self.env.get_runtime_config()})
        else:
            self._send(404, {"status": "error", "message": "not found"})

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n == 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw or b"{}")
        except Exception:
            return {}

    def do_POST(self):
        b = self.bridge
        parts = self.path.strip("/").split("/")
        if len(parts) == 1 and parts[0] == "config":
            cfg = self._body().get("config")
            if not isinstance(cfg, dict):
                self._send(400, {"status": "error",
                                 "message": "missing config object"})
                return
            try:
                self.env.set_runtime_config(cfg)
            except ValueError as e:
                self._send(400, {"status": "error", "message": str(e)})
                return
            except OSError as e:
                self._send(500, {"status": "error", "message": str(e)})
                return
            self._send(200, {"status": "ok"})
            return
        if len(parts) == 2 and parts[0] == "episode":
            cmd = parts[1]
            body = self._body()
            with b.lock:
                if cmd == "start":
                    if body.get("task"):
                        b.task = body["task"]
                    if body.get("instruction"):
                        b.instruction = body["instruction"]
                    b.running = True
                    b.paused = False
                    if not b.episode_active:
                        # Fresh start (or first connect after disconnect):
                        # begin a new episode via env.reset().
                        b.pending_reset = True
                        b.episode_active = False
                    # Resume from pause: episode_active is True, so keep the
                    # running episode — do NOT reset. The data loop resumes
                    # sending obs from where it paused, and the board keeps
                    # its step counter (see VlaServing paused_ handling).
                    self._send(200, {"status": "ok"})
                elif cmd == "stop":
                    b.running = False
                    self._send(200, {"status": "ok"})
                elif cmd == "reset":
                    b.pending_reset = True
                    b.episode_active = False
                    b.running = True
                    self._send(200, {"status": "ok"})
                elif cmd == "step":
                    b.paused = False
                    b.running = True
                    # one-step is approximated by letting the loop run once;
                    # board-side MVP maps step to "run loop"
                    self._send(200, {"status": "ok"})
                else:
                    self._send(400, {"status": "error",
                                     "message": f"unknown cmd {cmd}"})
            return
        self._send(404, {"status": "error", "message": "not found"})


def make_control_handler(bridge, env):
    ControlHandler.bridge = bridge
    ControlHandler.env = env
    return ControlHandler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_port", type=int, default=30005)
    ap.add_argument("--control_port", type=int, default=30006)
    ap.add_argument("--task", default="beat_block_hammer")
    ap.add_argument("--instruction", default="",
                    help="fixed task instruction; when empty, SimEnv "
                         "auto-generates an 'unseen' instruction per episode "
                         "like eval_policy.py")
    ap.add_argument("--pi0_step", type=int, default=50)
    ap.add_argument("--state_dim", type=int, default=14)
    ap.add_argument("--stub", action="store_true",
                    help="use StubSimEnv (no SAPIEN/RoboTwin deps)")
    ap.add_argument("--robotwin_root", default=None)
    ap.add_argument("--view_stride", type=int, default=1,
                    help="send a view-only frame every N action steps "
                         "(0=disable, 1=every step)")
    args = ap.parse_args()

    env = StubSimEnv() if args.stub else SimEnv(args.robotwin_root)
    if not args.stub:
        # RoboTwin env code resolves ./assets relative to the repo root, so
        # align the process cwd with the discovered root. This lets the bridge
        # run directly from a nested dir (e.g. RoboTwin/policy/pi0/sim_bridge).
        try:
            os.chdir(env.root)
        except OSError as e:
            print(f"[bridge] warning: cannot chdir to {env.root}: {e}",
                  flush=True)
    bridge = BridgeState()
    bridge.task = args.task
    bridge.instruction = args.instruction or ""
    bridge.view_stride = args.view_stride

    data_srv = _DataServer(("0.0.0.0", args.data_port),
                           make_data_handler(bridge, env))
    bridge.data_srv = data_srv  # handler can shutdown() to exit the process
    ctrl_srv = ThreadingHTTPServer(("0.0.0.0", args.control_port),
                                   make_control_handler(bridge, env))
    ctrl_srv.daemon_threads = True

    def _serve(srv, name):
        try:
            print(f"[bridge] {name} listening on {srv.server_address}",
                  flush=True)
            srv.serve_forever()
        except Exception as e:
            print(f"[bridge] {name} error: {e}", flush=True)
        finally:
            srv.shutdown()

    threading.Thread(target=_serve, args=(ctrl_srv, "control"),
                     daemon=True).start()
    # data server in main thread (one board client at a time)
    print(f"[bridge] data on 0.0.0.0:{args.data_port}  "
          f"control on 0.0.0.0:{args.control_port}  "
          f"stub={args.stub}  task={bridge.task}", flush=True)
    try:
        data_srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop_flag = True
        data_srv.shutdown()
        ctrl_srv.shutdown()
        env.close()


if __name__ == "__main__":
    main()
