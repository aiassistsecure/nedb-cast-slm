"""
cast.evaluate — grade a checkpoint with the REAL parser.

Three metrics, reported separately because they fail differently:

  valid_rate      — fraction of generations that parse at all
  exact_match     — fraction whose parsed plan is canonically equal to gold
                    (this is THE number; string equality is not required)
  per-clause      — exact-match broken down by which clause the gold plan uses,
                    so a clause failing badly cannot hide inside a good average

Held-out is scored with the same code but on phrasings absent from training, so
in-distribution vs held-out separates memorisation from generalisation. Both get
reported; never just the flattering one.
"""
from __future__ import annotations

import collections
import json
import os
import time
from typing import Dict, List, Optional

import torch

from nedb.query import parse_nql

from .model import CastConfig, CastModel
from .sampler import canonical
from .tokenizer import CastTokenizer


def load_checkpoint(ckpt_path: str, tokenizer_path: str):
    tok = CastTokenizer.load(tokenizer_path)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CastModel(CastConfig(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok, ck.get("step", -1)


@torch.no_grad()
def predict_batch(model: CastModel, tok: CastTokenizer, prompts: List[str],
                  max_new_tokens: int = 72, group_size: int = 24) -> List[str]:
    """Greedy-decode a batch of prompts.

    IMPORTANT — why this does not naively left-pad:

    The model uses LEARNED positional embeddings and no padding mask. If prompts
    of different lengths are padded into one tensor, the pad tokens shift every
    real token's position AND get attended to, which corrupts short prompts that
    share a batch with long ones. That produced a bizarre signature during
    development: exact-match accuracy *rose* with clause count (1-clause queries
    scored 0.9% while 4-clause scored 12.5%), because longer prompts needed less
    padding. It looked like a model capacity limit; it was an inference bug.

    Fix: group prompts of IDENTICAL encoded length so every batch is padding-free.
    This keeps batching's speed benefit while making batched output bit-identical
    to single-sequence decoding.
    """
    enc = [[tok.bos_id] + tok.encode(p) + [tok.sep_id] for p in prompts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out: List[Optional[str]] = [None] * len(enc)

    # bucket by exact length, then chunk each bucket
    buckets: Dict[int, List[int]] = {}
    for i in order:
        buckets.setdefault(len(enc[i]), []).append(i)

    for L, idxs in buckets.items():
        for s in range(0, len(idxs), group_size):
            grp = idxs[s:s + group_size]
            X = torch.tensor([enc[i] for i in grp], dtype=torch.long)
            done = [False] * len(grp)
            gen: List[List[int]] = [[] for _ in grp]
            cur = X
            for _ in range(max_new_tokens):
                ctx = cur[:, -model.cfg.block_size:]
                logits, _ = model(ctx)
                nxt = logits[:, -1, :].argmax(dim=-1)
                for j, t in enumerate(nxt.tolist()):
                    if done[j]:
                        continue
                    if t == tok.eos_id:
                        done[j] = True
                        continue
                    gen[j].append(t)
                cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
                if all(done):
                    break
            for j, i in enumerate(grp):
                out[i] = tok.decode(gen[j])
    return [o if o is not None else "" for o in out]


def evaluate(ckpt_path: str, data_path: str, tokenizer_path: str,
             limit: Optional[int] = 1000, batch_size: int = 32,
             verbose_failures: int = 0) -> Dict:
    model, tok, step = load_checkpoint(ckpt_path, tokenizer_path)
    rows = [json.loads(l) for l in open(data_path)]
    if limit:
        rows = rows[:limit]

    t0 = time.time()
    n_valid = 0
    n_exact = 0
    clause_tot: collections.Counter = collections.Counter()
    clause_ok: collections.Counter = collections.Counter()
    failures: List[Dict] = []

    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        preds = predict_batch(model, tok, [r["prompt"] for r in chunk])
        for r, pred in zip(chunk, preds):
            clauses = r.get("clauses") or []
            for c in clauses:
                clause_tot[c] += 1
            parsed = None
            try:
                parsed = parse_nql(pred)
                n_valid += 1
            except Exception:
                pass
            is_exact = parsed is not None and canonical(parsed) == r["plan"]
            if is_exact:
                n_exact += 1
                for c in clauses:
                    clause_ok[c] += 1
            elif len(failures) < verbose_failures:
                failures.append({"prompt": r["prompt"], "gold": r["nql"],
                                 "pred": pred,
                                 "reason": "invalid" if parsed is None else "mismatch"})

    n = len(rows)
    per_clause = {c: {"n": clause_tot[c], "exact": clause_ok[c],
                      "acc": round(clause_ok[c] / clause_tot[c], 4) if clause_tot[c] else None}
                  for c in sorted(clause_tot)}
    return {
        "step": step,
        "n": n,
        "valid_rate": round(n_valid / n, 4),
        "exact_match": round(n_exact / n, 4),
        "per_clause": per_clause,
        "eval_seconds": round(time.time() - t0, 1),
        "failures": failures,
        "data_path": data_path,
    }


def record_in_nedb(result: Dict, checkpoint_seq: int, db_path: str = "ledger",
                   eval_id: Optional[str] = None) -> Dict:
    """Write an eval result into NEDB, citing the checkpoint that produced it."""
    from nedb import NEDB
    db = NEDB(db_path)
    eid = eval_id or f"ev_{os.path.basename(result['data_path'])}_{result['step']}"
    existing = db.query(f'FROM evals WHERE _id = "{eid}"')
    if existing:
        return {"seq": existing[0].get("_seq"), "wrote": False, "head": db.head}
    doc = {
        "step": result["step"],
        "n": result["n"],
        "valid_rate": result["valid_rate"],
        "exact_match": result["exact_match"],
        "split": os.path.basename(result["data_path"]).replace(".jsonl", ""),
        "per_clause": json.dumps(result["per_clause"], sort_keys=True),
    }
    rec = db.put("evals", eid, doc, caused_by=[checkpoint_seq])
    return {"seq": rec["_seq"], "wrote": True, "head": db.head,
            "verify": db.verify()}
