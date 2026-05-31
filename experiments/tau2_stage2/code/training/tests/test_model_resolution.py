from pathlib import Path

from training.model_resolution import (
    from_pretrained_kwargs,
    resolve_model_source,
)


def _make_model_dir(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")


def test_resolve_model_source_uses_local_dir(monkeypatch, tmp_path):
    model_dir = tmp_path / "Qwen3.5-2B"
    _make_model_dir(model_dir)
    monkeypatch.setenv("EVOL_LOCAL_MODELS_DIR", str(tmp_path))

    source = resolve_model_source("Qwen/Qwen3.5-2B", "abc123")

    assert source.name == str(model_dir)
    assert source.revision is None
    assert source.local_path == str(model_dir)


def test_resolve_model_source_falls_back_to_hub(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOL_LOCAL_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("EVOL_REQUIRE_LOCAL_MODELS", raising=False)

    source = resolve_model_source("Qwen/Qwen3.5-2B", "abc123")

    assert source.name == "Qwen/Qwen3.5-2B"
    assert source.revision == "abc123"
    assert source.local_path is None


def test_require_local_models_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOL_LOCAL_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("EVOL_REQUIRE_LOCAL_MODELS", "1")

    try:
        resolve_model_source("Qwen/Qwen3.5-2B", "abc123")
    except FileNotFoundError as exc:
        assert "Qwen/Qwen3.5-2B" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_from_pretrained_kwargs_omits_revision_for_local(monkeypatch, tmp_path):
    model_dir = tmp_path / "Qwen3.5-2B"
    _make_model_dir(model_dir)
    monkeypatch.setenv("EVOL_LOCAL_MODELS_DIR", str(tmp_path))

    source = resolve_model_source("Qwen/Qwen3.5-2B", "abc123")
    kwargs = from_pretrained_kwargs(source, trust_remote_code=True)

    assert kwargs == {"trust_remote_code": True}
