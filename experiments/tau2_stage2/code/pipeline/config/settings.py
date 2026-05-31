"""Settings loaded from .env — supports CommonStack (required) and
OpenRouter (optional) inference providers. Provider selection is a
runtime concern handled by pipeline.runner; this module only loads
what's available.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # CommonStack — required. Primary first, fallbacks after, sourced from
    # COMMONSTACK_API_KEY + COMMONSTACK_API_KEY_2/_3/... in .env.
    commonstack_api_key: str
    commonstack_api_keys: tuple[str, ...]
    commonstack_base_url: str
    # OpenRouter — optional. None / () / None when env vars are unset.
    # Same primary + fallback shape as CommonStack.
    openrouter_api_key: str | None
    openrouter_api_keys: tuple[str, ...]
    openrouter_base_url: str | None
    data_dir: Path = Path("data")
    max_retries: int = 3
    retry_base_delay_s: float = 2.0


def load_settings() -> Settings:
    load_dotenv(override=False)
    cs_keys = _collect_api_keys("COMMONSTACK_API_KEY", required=True)
    or_keys = _collect_api_keys("OPENROUTER_API_KEY", required=False)
    or_base = (
        os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if or_keys
        else None
    )
    return Settings(
        commonstack_api_key=cs_keys[0],
        commonstack_api_keys=cs_keys,
        commonstack_base_url=os.environ.get(
            "COMMONSTACK_BASE_URL", "https://api.commonstack.ai/v1"
        ),
        openrouter_api_key=(or_keys[0] if or_keys else None),
        openrouter_api_keys=or_keys,
        openrouter_base_url=or_base,
        data_dir=Path(os.environ.get("EVOL_DATA_DIR", "data")),
    )


def _collect_api_keys(prefix: str, *, required: bool) -> tuple[str, ...]:
    """Read PREFIX + PREFIX_2..PREFIX_9 from env. Empty/whitespace-only
    fallback slots are skipped without breaking the chain. If `required`
    and the primary is missing, raises RuntimeError; otherwise returns ().
    """
    primary = os.environ.get(prefix, "").strip()
    if not primary:
        if required:
            raise RuntimeError(f"Missing env var: {prefix}")
        return ()
    keys = [primary]
    for i in range(2, 10):
        v = os.environ.get(f"{prefix}_{i}", "").strip()
        if v:
            keys.append(v)
    return tuple(keys)
