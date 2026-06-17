"""Resolve HF model ids to local directories for offline runs."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


LOCAL_MODEL_DIRNAMES = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3.5-2B": "Qwen3.5-2B",
    "Qwen/Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen/Qwen3.6-35B-A3B": "Qwen3.6-35B-A3B",
}


@dataclass(frozen=True)
class ModelSource:
    original_name: str
    name: str
    revision: str | None
    local_path: str | None = None


def _looks_like_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and (
        (path / "tokenizer_config.json").is_file()
        or (path / "tokenizer.json").is_file()
    )


def _candidate_dirs(root: Path, model_name: str) -> list[Path]:
    names = [LOCAL_MODEL_DIRNAMES.get(model_name), model_name.replace("/", "__")]
    if "/" in model_name:
        names.append(model_name.rsplit("/", 1)[-1])
    return [root / n for n in names if n]


def resolve_model_source(model_name: str, revision: str | None = None) -> ModelSource:
    """Return the source to pass to transformers ``from_pretrained``.

    If ``EVOL_LOCAL_MODELS_DIR`` (or ``LOCAL_MODELS_DIR``) is set, known Qwen
    model ids are resolved to subdirectories under that root. Local directories
    are used without a Hub revision, because a plain copied model directory is
    not a git repo and cannot resolve Hub commit ids.
    """
    local_root = os.environ.get("EVOL_LOCAL_MODELS_DIR") or os.environ.get(
        "LOCAL_MODELS_DIR"
    )
    if local_root:
        root = Path(local_root).expanduser()
        for candidate in _candidate_dirs(root, model_name):
            if _looks_like_model_dir(candidate):
                return ModelSource(
                    original_name=model_name,
                    name=str(candidate),
                    revision=None,
                    local_path=str(candidate),
                )
        if os.environ.get("EVOL_REQUIRE_LOCAL_MODELS") == "1":
            checked = ", ".join(str(p) for p in _candidate_dirs(root, model_name))
            raise FileNotFoundError(
                f"EVOL_REQUIRE_LOCAL_MODELS=1 but no local model directory was "
                f"found for {model_name}. Checked: {checked}"
            )

    return ModelSource(
        original_name=model_name,
        name=model_name,
        revision=revision,
        local_path=None,
    )


def from_pretrained_kwargs(source: ModelSource, **kwargs) -> dict:
    """Build kwargs for transformers ``from_pretrained`` calls."""
    out = dict(kwargs)
    if source.revision is not None:
        out["revision"] = source.revision
    return out
