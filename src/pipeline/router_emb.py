"""Deployable embedding-based router featurizer.

`EmbeddingRouter` wraps an LLM embedding (last-token, normalized) + a fitted
sklearn head, and re-embeds prompts at predict time. It lives in its own module
(not the train_router_simple __main__ script) so the joblib artifact unpickles
cleanly in collect_traces / run_e2e_ablation_simple.
"""
from __future__ import annotations


class EmbeddingRouter:
    def __init__(self, model_id: str, clf, max_len: int = 1024):
        self.model_id = model_id
        self.clf = clf
        self.max_len = max_len
        self._tok = None
        self._mdl = None
        self._dev = None

    def __getstate__(self):
        return {"model_id": self.model_id, "clf": self.clf, "max_len": self.max_len}

    def __setstate__(self, st):
        self.__dict__.update(st)
        self._tok = None
        self._mdl = None
        self._dev = None

    def _ensure(self):
        if self._mdl is None:
            import torch
            from transformers import AutoTokenizer, AutoModel
            self._tok = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
            self._dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._mdl = AutoModel.from_pretrained(
                self.model_id,
                dtype=torch.float16 if self._dev == "cuda" else torch.float32,
            ).to(self._dev).eval()

    def embed(self, prompts):
        import numpy as np
        import torch
        self._ensure()
        out = []
        with torch.no_grad():
            for i in range(0, len(prompts), 16):
                enc = self._tok(list(prompts[i:i + 16]), padding=True, truncation=True,
                                max_length=self.max_len, return_tensors="pt").to(self._dev)
                h = self._mdl(**enc).last_hidden_state[:, -1]
                h = torch.nn.functional.normalize(h, dim=-1)
                out.append(h.float().cpu().numpy())
        return np.vstack(out)

    def predict_proba(self, prompts):
        return self.clf.predict_proba(self.embed(prompts))

    def predict(self, prompts):
        return self.clf.predict(self.embed(prompts))
