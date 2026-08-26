"""Shared single-line eval progress for pi0 remote inference."""

from __future__ import annotations

_GREEN = "\033[92m"
_RED = "\033[91m"
_RESET = "\033[0m"


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return ""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}K"
    return f"{n / (1024 * 1024):.1f}M"


class EvalLiveProgress:
    """One live status line: CYCLE N - Step: X / Y [ SEND OK 12KB | RECEIVE OK 5KB ].

    Every ``*_render`` call overwrites the same terminal line (carriage-return
    + clear-line), so the screen never floods regardless of how many inference
    cycles run. One line persists per episode until ``reset_episode`` / ``flush``.
    """

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

    def _phase_tag(
        self,
        phase: str,
        ok: bool | None,
        active: bool,
        payload_bytes: int = 0,
    ) -> str:
        if ok is True:
            btag = f" {_fmt_bytes(payload_bytes)}" if payload_bytes else ""
            return f"{_GREEN}{phase} OK{btag}{_RESET}"
        if ok is False:
            return f"{_RED}{phase} FAIL{_RESET}"
        if active:
            return phase
        return ""

    def _render(self) -> None:
        if not self.enabled or not self.active:
            return
        step_text = (
            f"{_GREEN}{self.env_step} / {self.step_lim}{_RESET}"
            if self.step_lim
            else "? / ?"
        )
        tags: list[str] = []
        send_tag = self._phase_tag(
            "SEND", self._send_ok, self._active == "SEND", self._send_bytes
        )
        if send_tag:
            tags.append(send_tag)
        recv_tag = self._phase_tag(
            "RECEIVE", self._recv_ok, self._active == "RECEIVE", self._recv_bytes
        )
        if recv_tag:
            tags.append(recv_tag)
        inner = " | ".join(tags) if tags else "..."
        line = f"  CYCLE {self.cycle + 1} - Step: {step_text} [ {inner} ]"
        print(f"\r\033[K{line}", end="", flush=True)


eval_live = EvalLiveProgress()
