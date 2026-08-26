"""Shared single-line eval progress for pi0 remote inference."""

from __future__ import annotations

import re

_GREEN = "\033[92m"
_RED = "\033[91m"
_RESET = "\033[0m"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

# Fixed field widths keep the rendered line a constant length, so each
# redraw only rewrites the cells that actually changed (digits / status
# slots) instead of clearing and repainting the whole line. That kills the
# full-line flicker of a "\r\033[K<line>" full rewrite per frame.
_CYCLE_W = 5  # cycle counter digits
_STEP_W_MIN = 3  # step counter digits (widened to len(str(step_lim)))
_SLOT_W = 9  # status field width after "SEND " / "RECEIVE "


def _vis_len(s: str) -> int:
    """Visible length of ``s`` (ANSI color escapes stripped)."""
    return len(_ANSI_RE.sub("", s))


def _fmt_bytes(n: int) -> str:
    """Fixed 6-char right-aligned byte tag, e.g. ``  5.5K`` / ``675.3K``."""
    if n < 1024:
        s = f"{n}B"
    elif n < 1024 * 1024:
        s = f"{n / 1024:.1f}K"
    else:
        s = f"{n / (1024 * 1024):.1f}M"
    return f"{s:>6}"


class EvalLiveProgress:
    """One live status line:
    ``CYCLE N - Step: X / Y [ SEND OK 675.3K | RECEIVE OK 5.5K ]``.

    The line is repainted with a bare carriage return (no clear-line), and
    every field is fixed-width: padded cycle/step numbers, constant-width
    byte tags, and status slots that always render (dashes while pending,
    dots while in flight, OK/FAIL when done). The visible line length is
    therefore constant, so the terminal redraws only the few cells that
    changed instead of flashing the whole line. One line persists per
    episode until ``reset_episode`` / ``flush``.
    """

    _line_claimed = False  # some instance owns the terminal status line

    @classmethod
    def line_claimed(cls) -> bool:
        """True once any instance started rendering the live status line.

        Lets third-party step loops (RoboTwin ``_base_task``'s fallback
        ``step: N / M`` print) stay silent instead of clearing a line that
        a live renderer here owns.
        """
        return EvalLiveProgress._line_claimed

    def __init__(self) -> None:
        self.enabled = False
        self.active = False
        self.cycle = 0
        self.env_step = 0
        self.step_lim = 0
        self._send_ok: bool | None = None
        self._recv_ok: bool | None = None
        self._send_bytes: int = 0
        self._recv_bytes: int = 0
        self._active: str | None = None
        self._last_len = 0  # visible length of the line on screen

    def reset_episode(self) -> None:
        self.active = False
        self.cycle = 0
        self.env_step = 0
        self.step_lim = 0
        self._send_ok = None
        self._recv_ok = None
        self._send_bytes = 0
        self._recv_bytes = 0
        self._active = None

    def begin_infer_cycle(
        self,
        cycle: int,
        env_step: int | None,
        step_lim: int | None,
    ) -> None:
        if not self.enabled:
            return
        self.active = True
        EvalLiveProgress._line_claimed = True
        self.cycle = cycle
        self.env_step = 0 if env_step is None else env_step
        self.step_lim = 0 if step_lim is None else step_lim
        self._send_ok = None
        self._recv_ok = None
        self._send_bytes = 0
        self._recv_bytes = 0
        self._active = None

    def begin_send(self) -> None:
        self._active = "SEND"
        self._render()

    def complete_send(self, ok: bool, payload_bytes: int = 0) -> None:
        self._send_ok = ok
        self._send_bytes = payload_bytes
        self._active = None
        self._render()

    def begin_recv(self) -> None:
        self._active = "RECEIVE"
        self._render()

    def complete_recv(self, ok: bool, payload_bytes: int = 0) -> None:
        self._recv_ok = ok
        self._recv_bytes = payload_bytes
        self._active = None
        self._render()

    def update_step(self, env_step: int, step_lim: int | None = None) -> bool:
        if not self.enabled or not self.active:
            return False
        self.env_step = env_step
        if step_lim is not None:
            self.step_lim = step_lim
        self._render()
        return True

    def flush(self) -> None:
        print("\r\033[K", end="", flush=True)
        self._last_len = 0

    def _phase_slot(
        self,
        phase: str,
        ok: bool | None,
        active: bool,
        payload_bytes: int = 0,
    ) -> str:
        """Fixed-width phase slot; the status field is always _SLOT_W wide."""
        if ok is True:
            stat = f"OK {_fmt_bytes(payload_bytes)}"
            return f"{phase} {_GREEN}{stat}{_RESET}"
        if ok is False:
            stat = f"{'FAIL':<{_SLOT_W}}"
            return f"{phase} {_RED}{stat}{_RESET}"
        stat = "." * _SLOT_W if active else "-" * _SLOT_W
        return f"{phase} {stat}"

    def _render(self) -> None:
        if not self.enabled or not self.active:
            return
        if self.step_lim:
            w = max(_STEP_W_MIN, len(str(self.step_lim)))
            step_text = (
                f"{_GREEN}{self.env_step:>{w}} / {self.step_lim:>{w}}{_RESET}"
            )
        else:
            step_text = "? / ?"
        send_slot = self._phase_slot(
            "SEND", self._send_ok, self._active == "SEND", self._send_bytes
        )
        recv_slot = self._phase_slot(
            "RECEIVE", self._recv_ok, self._active == "RECEIVE", self._recv_bytes
        )
        line = (
            f"  CYCLE {self.cycle + 1:>{_CYCLE_W}} - Step: {step_text}"
            f" [ {send_slot} | {recv_slot} ]"
        )
        # Bare \r overwrite: with constant widths only the changed cells
        # redraw. If this line is somehow shorter than what is on screen
        # (e.g. '? / ?' -> digits, or a byte tag overflowing its width),
        # pad with spaces so no stale characters survive past the new end.
        vis = _vis_len(line)
        if vis < self._last_len:
            line = f"{line}{' ' * (self._last_len - vis)}"
        self._last_len = max(vis, self._last_len)
        print(f"\r{line}", end="", flush=True)


eval_live = EvalLiveProgress()
