"""Render a service's config, then ask the platform to (re)start it.

We deliberately don't supervise anything ourselves -- no fork/exec, no
restart-on-crash -- because that's reimplementing systemd badly. This module is
the delegation boundary, with two interchangeable backends: SystemdBackend
(EnvironmentFile + systemctl) and ComposeBackend (env_file + docker compose).
Same interface, so systemd -> compose -> k8s swaps one adapter.

Every backend takes dry_run=True to log what it would do and touch nothing.
"""
import logging
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from service_config import ServiceSpec

log = logging.getLogger("control.lifecycle")


@dataclass
class ServiceStatus:
    name: str
    active: bool
    detail: str = ""


def render_env(env: dict) -> str:
    """Deterministic KEY=VALUE env-file body (sorted; systemd & compose both read it)."""
    return "".join(f"{k}={env[k]}\n" for k in sorted(env))


class LifecycleBackend(ABC):
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    @abstractmethod
    def apply_config(self, spec: ServiceSpec, env: dict) -> None:
        """Render launch-time env for the platform; applies on the next (re)start."""

    @abstractmethod
    def start(self, spec: ServiceSpec) -> None: ...
    @abstractmethod
    def stop(self, spec: ServiceSpec) -> None: ...
    @abstractmethod
    def restart(self, spec: ServiceSpec) -> None: ...
    @abstractmethod
    def status(self, spec: ServiceSpec) -> ServiceStatus: ...

    @staticmethod
    def _validate(spec: ServiceSpec, env: dict) -> None:
        # Reject typo'd / disallowed knobs before they reach a config file.
        bad = set(env) - set(spec.params)
        if bad:
            raise ValueError(
                f"{spec.name}: not launch-time params {sorted(bad)}; allowed {list(spec.params)}")

    def _run(self, cmd: list) -> str:
        if self.dry_run:
            log.info("[dry-run] %s", " ".join(shlex.quote(c) for c in cmd))
            return ""
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            log.error("%s -> rc=%d %s", cmd[0], res.returncode, res.stderr.strip())
        return res.stdout


class SystemdBackend(LifecycleBackend):
    """EnvironmentFile + systemctl. `user=True` targets `systemctl --user`."""

    def __init__(self, user: bool = False, dry_run: bool = False):
        super().__init__(dry_run)
        self._scope = ["--user"] if user else []

    def apply_config(self, spec: ServiceSpec, env: dict) -> None:
        self._validate(spec, env)
        body = render_env(env)
        p = Path(spec.env_file)
        if self.dry_run:
            log.info("[dry-run] write %s:\n%s", p, body.rstrip())
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
        log.info("%s config staged (%d keys) -> applies on next restart", spec.name, len(env))

    def start(self, spec):   self._systemctl("start", spec.unit)
    def stop(self, spec):    self._systemctl("stop", spec.unit)
    def restart(self, spec): self._systemctl("restart", spec.unit)

    def status(self, spec) -> ServiceStatus:
        out = self._systemctl("is-active", spec.unit).strip() or "unknown"
        return ServiceStatus(spec.name, out == "active", out)

    def _systemctl(self, verb: str, unit: str) -> str:
        if self.dry_run and verb == "is-active":
            log.info("[dry-run] systemctl %s %s %s", " ".join(self._scope), verb, unit)
            return "active"
        return self._run(["systemctl", *self._scope, verb, unit])


class ComposeBackend(LifecycleBackend):
    """Dockerised: compose is the systemd-equivalent.

    env_file for config, `docker compose <verb>` for lifecycle. See README.md.
    """

    def __init__(self, compose_file: str, env_dir: str = "/etc/edge-ai/compose",
                 dry_run: bool = False):
        super().__init__(dry_run)
        self._base = ["docker", "compose", "-f", compose_file]
        self._env_dir = Path(env_dir)

    def apply_config(self, spec: ServiceSpec, env: dict) -> None:
        self._validate(spec, env)
        body = render_env(env)
        p = self._env_dir / f"{spec.compose_service}.env"
        if self.dry_run:
            log.info("[dry-run] write %s:\n%s", p, body.rstrip())
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)

    def start(self, spec):   self._run([*self._base, "up", "-d", spec.compose_service])
    def stop(self, spec):    self._run([*self._base, "stop", spec.compose_service])
    def restart(self, spec):
        self._run([*self._base, "up", "-d", "--force-recreate", spec.compose_service])

    def status(self, spec) -> ServiceStatus:
        out = self._run([*self._base, "ps", "--status=running", "--services"])
        active = spec.compose_service in out.split()
        return ServiceStatus(spec.name, active, "running" if active else "stopped")
