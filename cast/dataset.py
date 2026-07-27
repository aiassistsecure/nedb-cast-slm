"""
cast.dataset — generate the corpus, content-address it, record it in NEDB.

Two properties matter here:

  1. CONTENT-ADDRESSED. The run id is a BLAKE2b hash over the generator source,
     the config, and the resulting pairs. Regenerating with the same seed and
     unchanged code yields the same id, so re-running writes nothing. Change the
     paraphraser and the id changes — which is the point: the dataset manifest
     in NEDB always names exactly which generator produced it.

  2. EVERY PAIR IS PARSER-VERIFIED. No pair enters the corpus unless
     nedb.query.parse_nql round-trips it to a plan canonically equal to the
     sampled plan. A generator bug fails the build loudly instead of quietly
     teaching the model invalid syntax.

Splits are made by PLAN canonical form, not by row, so the same plan can never
appear in both train and eval. Held-out eval additionally uses phrasings absent
from the training pool, separating memorisation from generalisation.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from nedb.query import parse_nql

from . import grammar, paraphrase as para_mod, sampler as samp_mod
from .paraphrase import paraphrase
from .sampler import canonical, clauses_present, render_nql, sample_plan


def _source_fingerprint() -> str:
    """Hash the generator source so the dataset id changes when logic changes."""
    h = hashlib.blake2b(digest_size=16)
    for mod in (grammar, samp_mod, para_mod):
        path = mod.__file__
        with open(path, "rb") as fh:
            h.update(hashlib.blake2b(fh.read(), digest_size=16).digest())
    return h.hexdigest()


def generate(n_train: int = 200_000, n_eval: int = 4_000,
             n_holdout: int = 2_000, seed: int = 1337,
             out_dir: str = "data") -> Dict[str, Any]:
    """Generate train/eval/holdout splits. Returns a manifest dict."""
    rng = random.Random(seed)

    # ---- generate a pool of unique plans first, so splits are plan-disjoint
    #      (a plan appearing in both train and eval would inflate scores)
    target_plans = int((n_train + n_eval + n_holdout) * 0.75)
    plans: Dict[str, Tuple[Dict, Any, Any]] = {}
    guard = 0
    while len(plans) < target_plans and guard < target_plans * 40:
        guard += 1
        plan, dom, coll = sample_plan(rng)
        nql = render_nql(plan)
        # HARD GATE: must round-trip through the real parser
        try:
            parsed = parse_nql(nql)
        except Exception as e:
            raise RuntimeError(f"generator emitted unparseable NQL: {nql!r} -> {e}")
        cf = canonical(plan)
        if canonical(parsed) != cf:
            raise RuntimeError(f"canonical mismatch for {nql!r}")
        plans.setdefault(cf, (plan, dom, coll))

    keys = list(plans.keys())
    rng.shuffle(keys)

    n_eval_plans = max(1, int(len(keys) * 0.06))
    n_hold_plans = max(1, int(len(keys) * 0.04))
    # NOTE: these MUST stay lists, never sets. `list(set_of_strings)` iterates in
    # an order that depends on PYTHONHASHSEED, which is randomised per process —
    # that silently broke content-addressing (same seed, different run_id).
    eval_keys = keys[:n_eval_plans]
    hold_keys = keys[n_eval_plans:n_eval_plans + n_hold_plans]
    train_keys = keys[n_eval_plans + n_hold_plans:]

    def build(key_list, count, holdout=False):
        rows = []
        kl = list(key_list)
        if not kl:
            return rows
        i = 0
        while len(rows) < count:
            k = kl[i % len(kl)]
            i += 1
            plan, dom, coll = plans[k]
            prompt = paraphrase(plan, dom, coll, rng, holdout=holdout)
            rows.append({
                "prompt": prompt,
                "nql": render_nql(plan),
                "plan": k,
                "domain": dom.name,
                "coll": coll.name,
                "clauses": clauses_present(plan),
            })
        return rows

    train = build(train_keys, n_train)
    ev = build(eval_keys, n_eval)
    hold = build(hold_keys, n_holdout, holdout=True)

    # de-dup exact (prompt,plan) pairs inside train — repeated identical rows
    # waste steps without adding signal
    seen = set()
    dedup = []
    for r in train:
        sig = (r["prompt"], r["plan"])
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append(r)
    train = dedup

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, rows in (("train", train), ("eval", ev), ("holdout", hold)):
        p = os.path.join(out_dir, f"{name}.jsonl")
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        paths[name] = p

    # ---- content address: source fingerprint + config + data
    src_fp = _source_fingerprint()
    h = hashlib.blake2b(digest_size=16)
    h.update(src_fp.encode())
    h.update(json.dumps({"n_train": n_train, "n_eval": n_eval,
                         "n_holdout": n_holdout, "seed": seed},
                        sort_keys=True).encode())
    for name in ("train", "eval", "holdout"):
        with open(paths[name], "rb") as fh:
            h.update(hashlib.blake2b(fh.read(), digest_size=16).digest())
    run_id = "ds_" + h.hexdigest()

    manifest = {
        "run_id": run_id,
        "source_fingerprint": src_fp,
        "seed": seed,
        "unique_plans": len(plans),
        "train_rows": len(train),
        "eval_rows": len(ev),
        "holdout_rows": len(hold),
        "train_plans": len(train_keys),
        "eval_plans": len(eval_keys),
        "holdout_plans": len(hold_keys),
        "paths": paths,
        "clause_p": dict(samp_mod.CLAUSE_P),
        "domains": [d.name for d in grammar.DOMAINS],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def record_in_nedb(manifest: Dict[str, Any], db_path: str = "ledger") -> Dict[str, Any]:
    """Write the dataset manifest into NEDB. Idempotent by content address.

    Returns {"seq": int, "wrote": bool, "head": str, "verify": bool}.
    """
    from nedb import NEDB

    db = NEDB(db_path)
    existing = db.query(f'FROM datasets WHERE _id = "{manifest["run_id"]}"')
    if existing:
        return {"seq": existing[0].get("_seq"), "wrote": False,
                "head": db.head, "verify": db.verify()}

    doc = {k: v for k, v in manifest.items() if k != "clause_p"}
    doc["clause_p"] = json.dumps(manifest["clause_p"], sort_keys=True)
    rec = db.put("datasets", manifest["run_id"], doc)
    return {"seq": rec["_seq"], "wrote": True, "head": db.head,
            "verify": db.verify()}
