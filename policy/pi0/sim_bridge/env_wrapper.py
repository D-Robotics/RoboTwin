"""Environment wrapper for the VLA Sim Bridge.

Two implementations sharing one interface:
  - SimEnv: wraps a RoboTwin TASK_ENV (SAPIEN). Faithful to
    script/eval_policy.py setup sequence: class_decorator(task) -> setup_demo
    -> get_obs -> take_action -> close_env. Only import lazily so the file
    loads on a host without sapien (stub mode / mock board dev).
  - StubSimEnv: synthetic 224x224 RGB + fake state + moving marker, so the
    full Bridge <-> board <-> browser loop can run with zero sim deps.

Interface (both classes):
  list_tasks() -> [str]
  reset(task, seed=0, instruction="") -> None
  get_obs() -> (rgb_list:[HxWx3 uint8], state:[float], instruction:str)
    rgb_list is [head, left_wrist, right_wrist] matching the openpi
    AlohaInputs training order (base_0_rgb/left_wrist_0_rgb/
    right_wrist_0_rgb). Images are sent at raw resolution; the board
    letterboxes them to 224x224 (aspect-preserving + black pad), same as
    the eval.sh path.
  take_action(action: np.ndarray) -> None
  eval_success() -> bool
  close() -> None
"""

import os
import re
import sys
import time
import numpy as np


def _discover_robotwin_root(start):
    """Walk up from `start` looking for the RoboTwin repo root.

    The repo root is identified by its landmark files: script/eval_policy.py,
    envs/ (package) and task_config/. The bridge may live anywhere under the
    repo (e.g. RoboTwin/policy/pi0/sim_bridge), so resolve the root explicitly
    instead of assuming cwd == repo root.
    """
    cur = os.path.abspath(start)
    while True:
        if (os.path.isfile(os.path.join(cur, "script", "eval_policy.py"))
                and os.path.isdir(os.path.join(cur, "envs"))
                and os.path.isdir(os.path.join(cur, "task_config"))):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _resolve_robotwin_root(robotwin_root):
    """Return (root, verified). verified=False means no landmark-based root
    was found and the caller should print a --robotwin_root hint on failure."""
    if robotwin_root:
        # Absolutize NOW, before sim_bridge.main() chdirs: a relative root
        # must resolve against the launch cwd, not the post-chdir cwd.
        return os.path.abspath(robotwin_root), True
    here = os.path.dirname(os.path.abspath(__file__))
    for start in (os.getcwd(), here):
        found = _discover_robotwin_root(start)
        if found:
            return found, True
    return os.getcwd(), False


# Web-exposed runtime config: line-patch config.yaml so the Chinese comments
# and unrelated keys survive (a full yaml round-trip would destroy them).
_RUNTIME_CONFIG_KEYS = ("test_num", "sample", "rnd", "save_sample")
_RUNTIME_CONFIG_DEFAULTS = {"test_num": 100, "sample": 0, "rnd": False,
                            "save_sample": False}
_RUNTIME_CONFIG_FORMAT = {
    "test_num": lambda v: str(int(v)),
    "sample": lambda v: str(int(v)),
    "rnd": lambda v: "true" if v else "false",
    "save_sample": lambda v: "true" if v else "false",
}


def _validate_runtime_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("config must be an object")
    for key in ("test_num", "sample"):
        v = cfg.get(key)
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{key} must be an integer")
        lo, hi = (1, 10000) if key == "test_num" else (0, 10 ** 9)
        if not (lo <= v <= hi):
            raise ValueError(f"{key} out of range [{lo}, {hi}]")
    for key in ("rnd", "save_sample"):
        if not isinstance(cfg.get(key), bool):
            raise ValueError(f"{key} must be a boolean")


def read_runtime_config(path):
    """Return {test_num, sample, rnd, save_sample} from a config.yaml,
    falling back to defaults when the file is missing or unreadable.
    Never raises."""
    import yaml
    cfg = dict(_RUNTIME_CONFIG_DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return cfg
    for key in _RUNTIME_CONFIG_KEYS:
        if key in data:
            cfg[key] = data[key]
    return cfg


def write_runtime_config(path, cfg):
    """Validate and persist {test_num, sample, rnd, save_sample} into a
    config.yaml.

    Line-patches only the target key lines, preserving every other line
    (comments, unrelated keys). Missing keys are appended at the end.
    Raises ValueError on invalid input; OSError on write failure.
    """
    _validate_runtime_config(cfg)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out = []
    patched = set()
    for line in lines:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)[ \t]*:[ \t]*(.*?)([ \t]*)(#.*)?$",
                     line)
        if m and m.group(1) in _RUNTIME_CONFIG_KEYS:
            key = m.group(1)
            out.append(f"{key}: {_RUNTIME_CONFIG_FORMAT[key](cfg[key])}"
                       f"{m.group(3)}{m.group(4) or ''}\n")
            patched.add(key)
        else:
            out.append(line)
    for key in _RUNTIME_CONFIG_KEYS:
        if key not in patched:
            out.append(f"{key}: {_RUNTIME_CONFIG_FORMAT[key](cfg[key])}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)


class StubSimEnv:
    """Synthetic environment for x86 loop testing without SAPIEN/RoboTwin."""

    TASKS = ["beat_block_hammer", "blocks_ranking_rgb", "click_alarmclock"]

    step_lim = 400  # mirror TASK_ENV.step_lim budget for the stub loop

    def __init__(self):
        self.task = self.TASKS[0]
        self.instruction = ""
        self.step = 0
        self.episode = 0
        self._suc = 0
        self._total = 0
        self._reset_pending = True
        self._runtime_cfg = {"test_num": 0, "sample": 0, "rnd": False,
                            "save_sample": False}

    @property
    def take_action_cnt(self):
        return self.step

    @property
    def ep_num(self):
        return self.episode

    @property
    def test_num(self):
        # 0 = unbounded: the stub loop runs until the board disconnects.
        return self._runtime_cfg["test_num"]

    def get_runtime_config(self):
        return dict(self._runtime_cfg)

    def set_runtime_config(self, cfg):
        _validate_runtime_config(cfg)
        self._runtime_cfg = dict(cfg)

    def list_tasks(self):
        return list(self.TASKS)

    def reset(self, task=None, seed=0, instruction=""):
        if task:
            self.task = task
        self.instruction = instruction or f"demo instruction for {self.task}"
        self.step = 0
        self.episode += 1
        self._reset_pending = False

    def _frame(self, cam_idx):
        h = w = 224
        bg = np.full((h, w, 3), (10 + cam_idx * 20), dtype=np.uint8)
        t = self.step % 200
        cx = int(w * (0.1 + 0.8 * (t / 200.0)))
        cy = int(h * (0.5 + 0.3 * np.sin(t / 20.0)))
        r = 18
        color = [(220, 40, 40), (40, 220, 40), (40, 80, 220)][cam_idx]
        for y in range(max(0, cy - r), min(h, cy + r)):
            for x in range(max(0, cx - r), min(w, cx + r)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    bg[y, x] = color
        text = f"{self.task[:10]} c{cam_idx} s{self.step}"
        self._blit_text(bg, text, 4, 4 + cam_idx * 16)
        return bg

    @staticmethod
    def _blit_text(img, text, x, y):
        # Tiny 5x7 font for 0-9,a-z,_,space. Keeps stub dep-free.
        try:
            from PIL import Image, ImageDraw
            pil = Image.fromarray(img)
            ImageDraw.Draw(pil).text((x, y), text, fill=(255, 255, 255))
            np.copyto(img, np.array(pil))
        except Exception:
            pass

    def get_obs(self):
        self.step += 1
        rgb = [self._frame(i) for i in range(3)]
        state = np.full(14, float(self.step % 100) / 100.0, dtype=np.float64)
        return rgb, state, self.instruction

    def take_action(self, action):
        self.step += 1

    def eval_success(self):
        # Succeed every 200 steps so the demo shows occasional successes.
        return self.step > 0 and self.step % 200 == 0

    def close(self):
        pass


class SimEnv:
    """Wraps a real RoboTwin TASK_ENV. Run from the RoboTwin repo root.

    Faithful to script/eval_policy.py setup: `args` is the per-task
    `task_config/<task_config>.yml` (e.g. demo_clean) augmented with
    embodiment/camera wiring; setup_demo(**args); then get_obs/take_action.
    Deviations from eval_policy (intentional, for live demo):
      - eval_video_log forced False (no ffmpeg video recording).
      - expert_check / play_once seed validation skipped; when no
        instruction is given (caller/HTTP/CLI or TASK_ENV), a fresh
        'unseen' instruction is generated so
        the board never sees an empty prompt.
    """

    _SKIP = {"__init__", "_base_task", "_GLOBAL_CONFIGS"}

    def __init__(self, robotwin_root=None, task_config="demo_clean"):
        self.root, self._root_verified = _resolve_robotwin_root(robotwin_root)
        self.task_config = task_config
        self.task_env = None
        self.task = None
        self.instruction = ""
        self._args = None
        self._ep = 0
        self._test_num = 0  # total episodes from config.yaml; 0 until loaded

    def _import_helpers(self):
        import importlib
        import sys
        script_dir = os.path.join(self.root, "script")
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        try:
            return importlib.import_module("eval_policy")
        except ModuleNotFoundError as e:
            if not self._root_verified:
                raise RuntimeError(
                    f"could not locate the RoboTwin repo root (looked for "
                    f"script/eval_policy.py + envs/ + task_config/ near "
                    f"{self.root}); pass --robotwin_root <RoboTwin repo root>"
                ) from e
            raise

    def list_tasks(self):
        envs_dir = os.path.join(self.root, "envs")
        if not os.path.isdir(envs_dir):
            return ["beat_block_hammer"]
        out = []
        for f in sorted(os.listdir(envs_dir)):
            if not f.endswith(".py"):
                continue
            name = f[:-3]
            if name in self._SKIP or name.startswith("_"):
                continue
            out.append(name)
        return out

    def _build_args(self, task):
        ep = self._import_helpers()
        import yaml
        from envs import CONFIGS_PATH
        from eval_args import build_task_args
        args = build_task_args(task, self.task_config, task, root=self.root,
                               get_embodiment_config_fn=ep.get_embodiment_config,
                               default_embodiment=["aloha-agilex"])
        args["policy_name"] = "pi0"
        args["instruction_type"] = "unseen"
        args["eval_mode"] = True
        args["render_freq"] = 0
        args["eval_video_log"] = False
        args.pop("eval_video_save_dir", None)

        # Global config.yaml -> args["cfg"]; trim static_camera_list.
        with open(os.path.join(self.root, "task_config", "config.yaml"),
                  "r", encoding="utf-8") as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)
        args["cfg"] = data
        for side in ("left_embodiment_config", "right_embodiment_config"):
            scl = args[side].get("static_camera_list")
            if isinstance(scl, list) and len(scl) > 1:
                del scl[1]
        # eval_policy references eval_video_save_dir even with video off.
        import tempfile
        args.setdefault("eval_video_save_dir", tempfile.mkdtemp(prefix="vla_"))
        return args

    @property
    def runtime_config_path(self):
        return os.path.join(self.root, "task_config", "config.yaml")

    def get_runtime_config(self):
        return read_runtime_config(self.runtime_config_path)

    def set_runtime_config(self, cfg):
        write_runtime_config(self.runtime_config_path, cfg)

    def _episode_info(self):
        """Build the episode params dict that beat_block_hammer.play_once()
        would have stored in self.info["info"], without running the expert
        trajectory: {"{A}": "020_hammer/base0", "{a}": "left"/"right"}.
        Arm side is chosen from the block x pose (left if x<0), mirroring
        play_once(). Falls back to an empty dict if env isn't ready."""
        info = {}
        try:
            block = getattr(self.task_env, "block", None)
            if block is not None and hasattr(block, "get_functional_point"):
                p = block.get_functional_point(0, "pose").p
                info["{a}"] = "left" if p[0] < 0 else "right"
        except Exception:
            pass
        if self.task == "beat_block_hammer":
            info["{A}"] = "020_hammer/base0"
        return info

    def _generate_instruction(self):
        """Generate a fresh 'unseen' instruction:
        generate_episode_descriptions(task, [episode_info], max) then
        np.random.choice(results[0]["unseen"]). Falls back to a plain
        default if generation is impossible (env not ready / no templates)."""
        import random
        try:
            utils_dir = os.path.join(self.root, "description", "utils")
            if utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            from generate_episode_instructions import (
                generate_episode_descriptions)
            results = generate_episode_descriptions(
                self.task, [self._episode_info()], 1)
            if results and results[0].get("unseen"):
                return random.choice(results[0]["unseen"])
        except Exception as e:
            print(f"[env] instruction generation failed: {e}", flush=True)
        return self._default_instruction()

    def _default_instruction(self):
        """Plain fallback so the board never sees an empty prompt."""
        try:
            import json
            path = os.path.join(self.root, "description", "task_instruction",
                                f"{self.task}.json")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            desc = data.get("full_description", "")
            if desc:
                return desc
        except Exception:
            pass
        return "use the arm to complete the task"

    def reset(self, task=None, seed=0, instruction=""):
        ep = self._import_helpers()
        if task:
            self.task = task
        if instruction:
            self.instruction = instruction
        if self._args is None or self._args.get("task_name") != self.task:
            self._args = self._build_args(self.task)
        # config.yaml test_num bounds the episode count
        # (uses it as the loop limit; without it the bridge runs past the
        # last eval_data/<task>/<sample>/<N>_pos.pkl into FileNotFoundError).
        # Re-read the web-editable keys on every reset so edits take effect
        # on the next episode.
        rc = read_runtime_config(self.runtime_config_path)
        self._args["cfg"].update(rc)
        self._test_num = int(rc["test_num"])
        if self.task_env is None:
            self.task_env = ep.class_decorator(self.task)
        self.task_env.suc = getattr(self.task_env, "suc", 0)
        self.task_env.test_num = getattr(self.task_env, "test_num", 0)
        self.task_env.setup_demo(now_ep_num=self._ep, seed=seed,
                                 is_test=True, **self._args)
        # eval_policy calls _set_eval_video_ffmpeg(ffmpeg) after setup_demo;
        # take_action reads self.eval_video_ffmpeg(_spcam) so they must exist.
        try:
            self.task_env._set_eval_video_ffmpeg(None, None)
        except Exception:
            setattr(self.task_env, "eval_video_ffmpeg", None)
            setattr(self.task_env, "eval_video_ffmpeg_spcam", None)
        if not self.instruction:
            try:
                self.instruction = self.task_env.get_instruction() or ""
            except Exception:
                self.instruction = ""
        if not self.instruction:
            # Generate a per-episode "unseen" instruction
            # via generate_episode_descriptions(play_once info).
            # Bridge skips play_once, so rebuild the same info dict from the
            # env (arm from block position) and generate. This fixes the
            # empty "Task: " prompt that previously reached the board.
            self.instruction = self._generate_instruction()
        if self.instruction:
            try:
                self.task_env.set_instruction(instruction=self.instruction)
            except Exception:
                pass
        self._ep += 1

    @property
    def step_lim(self):
        return getattr(self.task_env, "step_lim", 0)

    @property
    def take_action_cnt(self):
        return getattr(self.task_env, "take_action_cnt", 0)

    @property
    def test_num(self):
        """Total episodes to run, from config.yaml ``test_num``.

        Stays 0 until the first reset() builds the per-task args (which
        loads config.yaml into args["cfg"]); the bridge treats 0 as
        "unbounded" so the stub / pre-config path keeps running.
        """
        return self._test_num

    @property
    def ep_num(self):
        """Current 0-based episode index — the value that will be passed as
        ``now_ep_num`` to the next ``setup_demo``. Equals resets done so far
        (incremented at the end of reset, after setup_demo consumed it)."""
        return self._ep

    def get_obs(self):
        obs = self.task_env.get_obs()
        o = obs["observation"]
        # Camera order must match the openpi AlohaInputs training order:
        # base_0_rgb(cam_high) -> left_wrist_0_rgb(cam_left_wrist) ->
        # right_wrist_0_rgb(cam_right_wrist), i.e. [head, LEFT, RIGHT].
        # The board maps wire order directly to view.images[i] (no reorder),
        # so sending [head, left, right] here matches the eval.sh path.
        rgb = [
            o["head_camera"]["rgb"],
            o["left_camera"]["rgb"],
            o["right_camera"]["rgb"],
        ]
        # Send RAW resolution; the board ResizeWithPadToBuffer letterboxes to
        # 224x224 (aspect-preserving + black pad), identical to eval.sh and to
        # openpi training (resize_with_pad). No local distortion.
        state = np.asarray(obs["joint_action"]["vector"], dtype=np.float64)
        return rgb, state, self.instruction

    def take_action(self, action):
        self.task_env.take_action(np.asarray(action))

    def eval_success(self):
        return bool(getattr(self.task_env, "eval_success", False))

    def close(self):
        if self.task_env is not None:
            try:
                self.task_env.close_env(clear_cache=True)
            except Exception:
                pass
            self.task_env = None
