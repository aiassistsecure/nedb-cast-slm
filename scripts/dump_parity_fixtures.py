#!/usr/bin/env python3
"""
dump_parity_fixtures.py — produce the ground-truth fixtures the Rust parity test
checks against, AND (re)export model.cast from the SAME checkpoint load, so the
weights in model.cast are bit-identical to the weights that produced the logits.

Why export + fixtures in one script?
------------------------------------
The v1 training checkpoint is written every 500 steps while training runs, so
its weights change over time. If model.cast were exported from one torch.load
and the fixtures from a *different* torch.load, they could describe different
weights and parity would fail for a spurious reason. This script loads the
checkpoint exactly once, exports model.cast from that in-memory state, and then
runs every fixture prompt through a model built from the *same* state_dict.

What each fixture records (per prompt)
--------------------------------------
  prompt        : the raw input string
  input_ids     : [bos] + encode(prompt) + [sep]   (the greedy-decode context)
  prompt_logits : full final-LayerNorm-then-head logit vector (vocab_size floats)
                  at the LAST position of input_ids — i.e. exactly one full
                  forward pass through the whole network. This is the strong
                  numerical check.
  gen_ids       : the greedy continuation token ids (no specials, EOS strips),
                  produced by the reference loop copied from cast.inference.
  nql           : the decoded NQL string (the exact user-facing output).

The Rust test asserts:
  * prompt_logits match within an absolute tolerance (default 1e-4)
  * gen_ids match EXACTLY
  * nql matches EXACTLY

Usage
-----
  python scripts/dump_parity_fixtures.py \
      --ckpt runs/v1/ckpt.pt --tokenizer data/tokenizer.json \
      --model-out model.cast --fixtures-out rust/nedb-cast-core/tests/fixtures.json \
      --max-new-tokens 72
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# make `import cast` and `import export_weights` work regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

# A fixed, hand-chosen prompt set. Deliberately exercises:
#   - LIMIT / numbers (single + multi-digit -> digit-merge detok)
#   - quoted string literals containing an NQL keyword ("rate limit")  [bug #2]
#   - dates with hyphens (VALID AS OF "2026-10-18")                    [bug #1]
#   - floats (31.0)                                                    [bug #1]
#   - multi-clause WHERE ... AND ...
#   - every kind of clause the grammar supports, spread across prompts
#   - a couple of short, easy ones (single clause)
PROMPTS = [
    "top 5 stylists in winter park",
    "paid orders over $99, newest first",
    "show me all services",
    "customers with lifetime value above 400000",
    "invoices that cleared, cheapest first, limit 10",
    "events search \"rate limit\" then trace caused_by reverse",
    "products valid as of 2026-10-18 ordered by title",
    "transactions where amount < 31.0 and sender = marisa",
    "wallets where balance < 2942.0 and label = governance",
    "blocks above height 400000",
    "appointments booked_at after 2025-01-02 limit 3",
    "leads grouped by stage count",
    "runs where loss < 0.05 order by step desc",
    "deploys where status = confirmed and latency < 200",
    "checkpoints traverse reviewed limit 7",
    "list orders where total >= 250",
    "give me the first 5 customers",
    "search \"no show\" in appointments",
    "crm leads owned by porter",
    "products where price <= 19.99 order by price asc limit 20",
]


def build_model(ck, itos):
    import torch
    from cast.model import CastConfig, CastModel

    cfg = CastConfig(**ck["cfg"])
    if cfg.vocab_size != len(itos):
        raise SystemExit(
            f"vocab mismatch: cfg.vocab_size={cfg.vocab_size} vs "
            f"tokenizer {len(itos)} tokens")
    model = CastModel(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def greedy_decode(model, tok, cfg, input_ids, max_new_tokens):
    """Reference greedy loop, mirroring cast.inference.cast_many for one prompt
    (batch of 1, therefore never any padding). Returns (gen_ids, prompt_logits).

    prompt_logits is the full logit vector at the last position of the ORIGINAL
    input_ids (the very first forward pass), captured before any token is added.
    """
    import torch

    cur = torch.tensor([input_ids], dtype=torch.long)
    gen = []
    prompt_logits = None
    with torch.no_grad():
        for stepi in range(max_new_tokens):
            ctx = cur[:, -cfg.block_size:]
            logits, _ = model(ctx)
            last = logits[:, -1, :]
            if stepi == 0:
                prompt_logits = last[0].detach().cpu().tolist()
            nxt = last.argmax(dim=-1)
            t = int(nxt.item())
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
            if t == tok.eos_id:
                break
            gen.append(t)
    return gen, prompt_logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v1/ckpt.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--model-out", default="model.cast")
    ap.add_argument(
        "--fixtures-out",
        default="rust/nedb-cast-core/tests/fixtures.json")
    ap.add_argument("--max-new-tokens", type=int, default=72)
    ap.add_argument("--logits-tol", type=float, default=1e-4)
    args = ap.parse_args()

    import export_weights
    from cast.tokenizer import CastTokenizer

    # -- single load of the (moving) checkpoint -------------------------------
    ck = export_weights.load_checkpoint(args.ckpt)
    with open(args.tokenizer) as fh:
        itos = json.load(fh)["itos"]

    # -- export model.cast from THIS exact state ------------------------------
    exp_info = export_weights.export_from_loaded(
        ck, itos, args.model_out, ckpt_path=args.ckpt)
    print(f"[fixtures] exported {args.model_out} "
          f"(step {exp_info['source_step']}, checksum "
          f"{exp_info['checksum_fnv1a64']})")

    # -- build the same model in torch and run the fixtures -------------------
    model, cfg = build_model(ck, itos)
    tok = CastTokenizer(itos)

    fixtures = []
    for p in PROMPTS:
        input_ids = [tok.bos_id] + tok.encode(p) + [tok.sep_id]
        gen_ids, prompt_logits = greedy_decode(
            model, tok, cfg, input_ids, args.max_new_tokens)
        nql = tok.decode(gen_ids)
        fixtures.append({
            "prompt": p,
            "input_ids": input_ids,
            "prompt_logits": prompt_logits,
            "gen_ids": gen_ids,
            "nql": nql,
        })

    out = {
        "meta": {
            "producer": "scripts/dump_parity_fixtures.py",
            "source_ckpt": args.ckpt,
            "source_step": exp_info["source_step"],
            "model_checksum_fnv1a64": str(exp_info["checksum_fnv1a64"]),
            "vocab_size": cfg.vocab_size,
            "logits_tol": args.logits_tol,
            "max_new_tokens": args.max_new_tokens,
            "note": ("prompt_logits is the full logit vector at the last "
                     "position of input_ids (one full forward pass)."),
        },
        "fixtures": fixtures,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.fixtures_out)), exist_ok=True)
    with open(args.fixtures_out, "w") as fh:
        json.dump(out, fh)
    print(f"[fixtures] wrote {len(fixtures)} fixtures to {args.fixtures_out}")
    # a quick human-readable peek
    for fx in fixtures[:4]:
        print(f"    {fx['prompt']!r}\n      -> {fx['nql']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
