"""BERT-based prompt router."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


@dataclass
class BertRouterConfig:
    """Configuration needed to train and reload a router."""

    base_model: str = "google/bert_uncased_L-2_H-128_A-2"
    max_length: int = 256
    small_model: str = "deepseek/deepseek-v3.2"
    large_model: str = "openai/gpt-5.4-2026-03-05"
    threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BertRouterConfig":
        allowed = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BertRouterConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class BertRouter:
    """Thin wrapper around AutoModelForSequenceClassification.

    Label convention:
      0 -> try small model first
      1 -> route directly to large model
    """

    config_filename = "router_config.json"

    def __init__(self, model: Any, tokenizer: Any, config: BertRouterConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    def from_pretrained_base(cls, config: BertRouterConfig) -> "BertRouter":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model,
            num_labels=2,
            id2label={0: "SMALL", 1: "LARGE"},
            label2id={"SMALL": 0, "LARGE": 1},
        )
        return cls(model=model, tokenizer=tokenizer, config=config)

    @classmethod
    def from_scratch(
        cls,
        config: BertRouterConfig,
        vocab_tokens: Sequence[str],
        work_dir: str | Path,
    ) -> "BertRouter":
        """Initialize a tiny BERT router without downloading a HF checkpoint."""
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        vocab_file = work_dir / "vocab.txt"
        with open(vocab_file, "w") as f:
            for token in vocab_tokens:
                f.write(f"{token}\n")

        tokenizer = BertTokenizer(vocab_file=str(vocab_file), do_lower_case=True)
        model_config = BertConfig(
            vocab_size=len(vocab_tokens),
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=max(512, config.max_length + 2),
            num_labels=2,
            id2label={0: "SMALL", 1: "LARGE"},
            label2id={"SMALL": 0, "LARGE": 1},
        )
        model = BertForSequenceClassification(model_config)
        return cls(model=model, tokenizer=tokenizer, config=config)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "BertRouter":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        config = BertRouterConfig.load(path / cls.config_filename)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        if device:
            model.to(torch.device(device))
        model.eval()
        return cls(model=model, tokenizer=tokenizer, config=config)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.config.save(path / self.config_filename)

    def predict_proba(
        self,
        prompts: list[str],
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> list[float]:
        """Return P(route_to_large) for each prompt."""
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(device)
        self.model.to(torch_device)
        self.model.eval()

        probs: list[float] = []
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start:start + batch_size]
                inputs = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(torch_device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits
                batch_probs = torch.softmax(logits, dim=-1)[:, 1]
                probs.extend(batch_probs.detach().cpu().tolist())
        return probs
