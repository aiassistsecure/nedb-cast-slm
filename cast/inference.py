"""
cast.inference — the public API. Load a checkpoint, cast prompts into plans.

Design notes worth knowing before you edit this file:

  * Greedy decoding by default. This is a DSL, not prose — there is exactly one
    right answer, so sampling can only hurt. temperature is exposed for
    experiments, not for production.

  * `cast()` returns NQL text. `plan()` returns the parsed plan dict and raises
    on invalid output. Studio wants `plan()`: a plan dict can be shown to a user
    as an editable preview before anything touches the database.

  * NO PADDING, EVER, IN A BATCH OF MIXED LENGTHS. The model has learned
    positional embeddings and no pad mask, so padding shifts positions and
    corrupts short sequences. `cast_many()` groups by identical encoded length
    so batches are padding-free. This bug cost us a 55-point accuracy swing
    during development — see docs/LORE.md.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch

from .model import CastConfig, CastModel
from .tokenizer import CastTokenizer


class Cast:
    """A loaded nedb-cast-slm model."""

    def __init__(self, model: CastModel, tokenizer: CastTokenizer,
                 step: int = -1, meta: Optional[Dict] = None):
        self.model = model
        self.tok = tokenizer
        self.step = step
        self.meta = meta or {}
        self.model.eval()

    # ------------------------------------------------------------------ load
    @classmethod
    def from_pretrained(cls, path: str) -> "Cast":
        """Load from a run directory (containing ckpt.pt) or an explicit .pt file.

        Looks for the tokenizer next to the checkpoint, then in ../data, then in
        ./data — so both `runs/v1` layouts and packaged layouts work.
        """
        if os.path.isdir(path):
            ckpt_path = os.path.join(path, "ckpt.pt")
        else:
            ckpt_path = path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"no checkpoint at {ckpt_path}")

        base = os.path.dirname(os.path.abspath(ckpt_path))
        candidates = [
            os.path.join(base, "tokenizer.json"),
            os.path.join(base, "..", "data", "tokenizer.json"),
            os.path.join(os.getcwd(), "data", "tokenizer.json"),
        ]
        tok_path = next((c for c in candidates if os.path.exists(c)), None)
        if tok_path is None:
            raise FileNotFoundError(
                "tokenizer.json not found next to the checkpoint or in ./data")

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = CastModel(CastConfig(**ck["cfg"]))
        model.load_state_dict(ck["model"])
        tok = CastTokenizer.load(tok_path)
        return cls(model, tok, step=ck.get("step", -1),
                   meta={"ckpt": ckpt_path, "tokenizer": tok_path})

    # ------------------------------------------------------------------ cast
    @torch.no_grad()
    def cast(self, prompt: str, max_new_tokens: int = 72,
             temperature: float = 0.0) -> str:
        """Cast one short prompt into NQL text."""
        return self.cast_many([prompt], max_new_tokens=max_new_tokens,
                              temperature=temperature)[0]

    @torch.no_grad()
    def cast_many(self, prompts: List[str], max_new_tokens: int = 72,
                  temperature: float = 0.0, group_size: int = 24) -> List[str]:
        """Cast many prompts. Batches only same-length inputs (see module docstring)."""
        enc = [[self.tok.bos_id] + self.tok.encode(p) + [self.tok.sep_id]
               for p in prompts]
        out: List[Optional[str]] = [None] * len(enc)

        buckets: Dict[int, List[int]] = {}
        for i, e in enumerate(enc):
            buckets.setdefault(len(e), []).append(i)

        for _, idxs in buckets.items():
            for s in range(0, len(idxs), group_size):
                grp = idxs[s:s + group_size]
                cur = torch.tensor([enc[i] for i in grp], dtype=torch.long)
                done = [False] * len(grp)
                gen: List[List[int]] = [[] for _ in grp]
                for _ in range(max_new_tokens):
                    ctx = cur[:, -self.model.cfg.block_size:]
                    logits, _ = self.model(ctx)
                    logits = logits[:, -1, :]
                    if temperature <= 0:
                        nxt = logits.argmax(dim=-1)
                    else:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        nxt = torch.multinomial(probs, 1).squeeze(-1)
                    for j, t in enumerate(nxt.tolist()):
                        if done[j]:
                            continue
                        if t == self.tok.eos_id:
                            done[j] = True
                            continue
                        gen[j].append(t)
                    cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
                    if all(done):
                        break
                for j, i in enumerate(grp):
                    out[i] = self.tok.decode(gen[j])
        return [o if o is not None else "" for o in out]

    # ------------------------------------------------------------------ plan
    def plan(self, prompt: str, **kw) -> Dict[str, Any]:
        """Cast to NQL, then parse with the REAL engine parser.

        Raises SyntaxError if the model produced something the engine cannot run.
        That is deliberate: better a loud failure than a silent wrong query.
        """
        from nedb.query import parse_nql
        nql = self.cast(prompt, **kw)
        try:
            return parse_nql(nql)
        except Exception as e:
            raise SyntaxError(f"model produced invalid NQL: {nql!r} ({e})") from e

    def try_plan(self, prompt: str, **kw) -> Dict[str, Any]:
        """Non-raising variant. Returns {ok, nql, plan|error}."""
        from nedb.query import parse_nql
        nql = self.cast(prompt, **kw)
        try:
            return {"ok": True, "nql": nql, "plan": parse_nql(nql)}
        except Exception as e:
            return {"ok": False, "nql": nql, "error": str(e)}

    # ------------------------------------------------------------------ run
    def run(self, prompt: str, db, **kw) -> List[dict]:
        """Cast and execute against a live NEDB database handle.

        `db` is an open `nedb.NEDB` instance. The plan is executed with the
        engine's own executor, so semantics are identical to a hand-written query.
        """
        return db.execute(self.plan(prompt, **kw))

    def __repr__(self) -> str:
        p = self.model.n_params() / 1e6
        return f"<Cast {p:.2f}M params, vocab {len(self.tok)}, step {self.step}>"
