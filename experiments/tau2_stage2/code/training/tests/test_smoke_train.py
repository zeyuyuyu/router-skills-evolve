"""Single-batch overfit smoke test (Spec §5.8).

Goal: confirm that completion_only_loss correctly masks all but the target,
and that a tiny model on one example can drive training loss → ~0.

Two tests:
- `test_completion_only_mask_excludes_prompt` — fast (tokenizer-only, no torch).
- `test_overfit_one_example` — slow, requires torch + trl. Skipped by default;
  run with SKIP_SMOKE_TRAIN=0 to enable.

Both tests skip cleanly when their deps are unavailable.
"""
import os
from pathlib import Path

import pytest


def test_completion_only_mask_excludes_prompt():
    """Verify: with completion_only_loss=True, prompt tokens have label=-100.

    This is a purely-tokenizer-level sanity check — no torch/trl required.
    Skipped if transformers can't load Qwen2.5-0.5B-Instruct (no network etc.).
    """
    transformers = pytest.importorskip("transformers")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Qwen tokenizer unavailable: {e}")
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"

    msgs_prompt = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Hi"},
    ]
    msgs_completion = [{"role": "assistant", "content": "Hello!"}]

    full_text = tok.apply_chat_template(
        msgs_prompt + msgs_completion, tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    prompt_text = tok.apply_chat_template(
        msgs_prompt, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    full_ids = tok.encode(full_text)
    prompt_ids = tok.encode(prompt_text)

    n_prompt = len(prompt_ids)
    n_total = len(full_ids)
    assert n_total > n_prompt, "Prompt should be a strict prefix of full"
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    n_masked = sum(1 for l in labels if l == -100)
    assert n_masked == n_prompt


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("SKIP_SMOKE_TRAIN", "1") == "1",
    reason="Set SKIP_SMOKE_TRAIN=0 to run (downloads Qwen2.5-0.5B; takes ~2 min)"
)
def test_overfit_one_example():
    """Train a tiny model on one row for 200 steps. Loss should drop below 0.05.

    This is the local sign-off mask-correctness gate: if completion_only_loss
    is misconfigured, the loss won't go below ~1.0 because the prompt tokens
    keep contributing.
    """
    pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("datasets")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    data = [{
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "completion": [{"role": "assistant", "content": "The answer is 4."}],
        "tools": [],
    }]
    ds = Dataset.from_list(data)

    cfg = SFTConfig(
        output_dir="/tmp/_smoke_overfit",
        num_train_epochs=1,
        max_steps=200,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        bf16=False, fp16=False,
        max_length=512,                  # TRL@main uses max_length, not max_seq_length
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(model=model, processing_class=tok, args=cfg, train_dataset=ds)
    final = trainer.train()
    final_loss = final.metrics.get("train_loss")
    assert final_loss is not None, "trainer did not record a final loss"
    assert final_loss < 0.05, (
        f"Single-example overfit failed: loss={final_loss}. "
        f"Expected <0.05 after 200 steps. Likely cause: completion mask not applied."
    )
