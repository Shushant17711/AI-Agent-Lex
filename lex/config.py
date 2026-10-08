"""Settings. Precedence: defaults < ~/.config/lex/config.toml < <workspace>/lex.toml < env vars < CLI flags."""

from __future__ import annotations

import os
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from lex.approval import MODES

DEFAULT_MODELS = {"gemini": "gemini-3.5-flash", "openai": "gpt-5-mini"}


@dataclass(frozen=True)
class Settings:
    provider: str = "gemini"
    model: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = ""
    temperature: float | None = None
    max_turns: int = 40  # tool-calling turns per agent per phase
    review_rounds: int = 2  # how many times the reviewer may send work back
    plan: bool = True
    review: bool = True
    approve_writes: str = "ask"
    approve_commands: str = "ask"
    command_timeout: int = 120

    def __post_init__(self) -> None:
        if not self.model:
            object.__setattr__(self, "model", DEFAULT_MODELS.get(self.provider, ""))
        for name in ("approve_writes", "approve_commands"):
            if getattr(self, name) not in MODES:
                raise ValueError(f"{name} must be one of {', '.join(MODES)}")

    def api_key(self) -> str:
        names = [self.api_key_env] if self.api_key_env else (
            ["GEMINI_API_KEY", "GOOGLE_API_KEY"] if self.provider == "gemini" else ["OPENAI_API_KEY"])
        return next((os.environ[n] for n in names if os.environ.get(n)), "")

    def public(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["has_api_key"] = bool(self.api_key())
        return d

    def update(self, **overrides: Any) -> "Settings":
        clean = {k: v for k, v in overrides.items() if v is not None and k in _FIELD_NAMES}
        if "provider" in clean and "model" not in clean and clean["provider"] != self.provider:
            clean["model"] = ""
        return replace(self, **clean)


_FIELD_NAMES = {f.name for f in fields(Settings)}

ENV_MAP = {
    "LEX_PROVIDER": "provider", "LEX_MODEL": "model", "LEX_BASE_URL": "base_url",
    "LEX_APPROVE_WRITES": "approve_writes", "LEX_APPROVE_COMMANDS": "approve_commands",
}


def load_dotenv(path: Path) -> None:
    """Minimal .env loader; never overrides variables that are already set."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    return {k.replace("-", "_"): v for k, v in data.items() if k.replace("-", "_") in _FIELD_NAMES}


def config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "lex"


def load_settings(workspace: Path, **cli: Any) -> Settings:
    load_dotenv(workspace / ".env")
    load_dotenv(config_home() / ".env")
    values: dict[str, Any] = {}
    values.update(_read_toml(config_home() / "config.toml"))
    values.update(_read_toml(workspace / "lex.toml"))
    values.update({field: os.environ[env] for env, field in ENV_MAP.items() if os.environ.get(env)})
    base = Settings()
    if "provider" in values and "model" not in values:
        values["model"] = ""
    return base.update(**values).update(**cli)
