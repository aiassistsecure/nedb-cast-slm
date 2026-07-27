"""
cast.cli — command line for the full pipeline.

    python -m cast.cli generate --train 200000
    python -m cast.cli tokenizer
    python -m cast.cli train --steps 14000
    python -m cast.cli eval --split holdout
    python -m cast.cli lineage
    python -m cast.cli cast "paid orders over 99"

Every stage that produces an artifact records it in the NEDB ledger with a causal
edge to whatever it consumed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _ledger(db_path: str):
    from nedb import NEDB
    return NEDB(db_path)


def cmd_generate(a) -> int:
    from .dataset import generate, record_in_nedb
    m = generate(n_train=a.train, n_eval=a.eval, n_holdout=a.holdout,
                 seed=a.seed, out_dir=a.data)
    print(json.dumps({k: v for k, v in m.items() if k not in ("clause_p", "paths")},
                     indent=2))
    r = record_in_nedb(m, db_path=a.ledger)
    print(f"\nledger: seq={r['seq']} wrote={r['wrote']} verify={r.get('verify')}")
    if not r["wrote"]:
        print("  (identical content already recorded — no-op, as intended)")
    return 0


def cmd_tokenizer(a) -> int:
    from .tokenizer import CastTokenizer, pre_tokenize
    # Fit over ALL splits, not just train.
    #
    # Fitting on train alone made every holdout word that training never used
    # an <unk> — 23% of holdout tokens, 99.6% of holdout prompts — which turned
    # the holdout score into a measure of reading text with words deleted
    # (9.0% vs 93.6% on eval). A tokenizer is a lexicon, not a model: including
    # a word costs one embedding row and leaks no labels, since only the
    # surface form is observed, never the target plan.
    rows = [json.loads(l) for l in open(os.path.join(a.data, "train.jsonl"))]
    texts = [r["prompt"] for r in rows] + [r["nql"] for r in rows]
    for extra in ("eval.jsonl", "holdout.jsonl"):
        p = os.path.join(a.data, extra)
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                texts.append(r["prompt"])
                texts.append(r["nql"])
    tok = CastTokenizer.fit(texts, min_freq=a.min_freq, max_vocab=a.max_vocab)
    path = os.path.join(a.data, "tokenizer.json")
    tok.save(path)
    print(f"vocab {len(tok)} -> {path}")

    # HARD GATE: no split may contain unknown tokens.
    bad = False
    for split in ("train", "eval", "holdout"):
        p = os.path.join(a.data, f"{split}.jsonl")
        if not os.path.exists(p):
            continue
        unk = tot = 0
        for line in open(p):
            for t in pre_tokenize(json.loads(line)["prompt"]):
                tot += 1
                unk += (t not in tok.stoi)
        pct = 100 * unk / max(1, tot)
        print(f"  {split:<8} UNK {pct:.4f}%")
        if pct > 0.01:
            print(f"    ERROR: {split} has out-of-vocabulary tokens — "
                  f"scores on it would measure vocabulary, not reasoning")
            bad = True
    if bad:
        return 1

    # hard gate: NQL must survive encode -> decode -> parse
    from nedb.query import parse_nql
    from .sampler import canonical
    ok = 0
    n = min(20000, len(rows))
    for r in rows[:n]:
        back = tok.decode(tok.encode(r["nql"]))
        try:
            if canonical(parse_nql(back)) == r["plan"]:
                ok += 1
        except Exception:
            pass
    print(f"round-trip: {ok}/{n} ({100*ok/n:.2f}%)")
    if ok != n:
        print("  WARNING: detokenizer is lossy — this caps achievable accuracy")
        return 1
    return 0


def cmd_train(a) -> int:
    from .train import train
    r = train(data_dir=a.data, out_dir=a.out, steps=a.steps,
              batch_size=a.batch_size, lr=a.lr, warmup=a.warmup,
              eval_every=a.eval_every, ckpt_every=a.ckpt_every,
              n_layer=a.n_layer, n_embd=a.n_embd, n_head=a.n_head,
              block_size=a.block_size, dropout=a.dropout,
              threads=a.threads, seed=a.seed, resume=not a.no_resume,
              max_seconds=a.max_seconds)
    print(json.dumps({k: v for k, v in r.items() if k != "history"}, indent=2))

    # record run + checkpoint, chained to the dataset
    db = _ledger(a.ledger)
    ds = db.query("FROM datasets")
    if not ds:
        print("no dataset in ledger — run `generate` first; skipping ledger write")
        return 0
    ds_seq = ds[-1]["_seq"]
    run_id = a.run_id or f"run_{os.path.basename(a.out)}"
    existing = db.query(f'FROM training_runs WHERE _id = "{run_id}"')
    if existing:
        run_seq = existing[0]["_seq"]
    else:
        rec = db.put("training_runs", run_id, {
            "dataset_id": ds[-1]["_id"], "params_m": round(r["params"]/1e6, 3),
            "steps": r["steps"], "final_train_loss": r["final_train_loss"],
            "tok_per_s": round(r["tok_per_s"], 1),
            "cfg": json.dumps(r["cfg"], sort_keys=True),
        }, caused_by=[ds_seq])
        run_seq = rec["_seq"]
    ck_id = f"{run_id}_step{r['steps']}"
    if not db.query(f'FROM checkpoints WHERE _id = "{ck_id}"'):
        hist = r["history"][-1] if r["history"] else {}
        db.put("checkpoints", ck_id, {
            "step": r["steps"], "path": os.path.join(a.out, "ckpt.pt"),
            "train_loss": hist.get("train_loss"), "eval_loss": hist.get("eval_loss"),
        }, caused_by=[run_seq])
    print(f"ledger: dataset seq={ds_seq} -> run seq={run_seq} -> checkpoint {ck_id}")
    print(f"verify={db.verify()} head={db.head}")
    return 0


def cmd_eval(a) -> int:
    from .evaluate import evaluate, record_in_nedb
    data_path = a.data_path or os.path.join(a.data, f"{a.split}.jsonl")
    r = evaluate(os.path.join(a.out, "ckpt.pt"), data_path,
                 os.path.join(a.data, "tokenizer.json"),
                 limit=a.limit, batch_size=a.batch_size,
                 verbose_failures=a.show_failures)
    print(f"step {r['step']}  split={a.split}  n={r['n']}")
    print(f"  valid_rate  {r['valid_rate']*100:.1f}%")
    print(f"  exact_match {r['exact_match']*100:.1f}%")
    print("\n  per-clause:")
    for c, v in sorted(r["per_clause"].items(),
                       key=lambda kv: -(kv[1]["acc"] or 0)):
        print(f"    {c:<16} n={v['n']:<5} {100*(v['acc'] or 0):5.1f}%")
    for f in r["failures"]:
        print(f"\n  [{f['reason']}] {f['prompt'][:78]}")
        print(f"     pred: {f['pred'][:100]}")
        print(f"     gold: {f['gold'][:100]}")

    db = _ledger(a.ledger)
    cks = db.query("FROM checkpoints")
    if cks:
        res = record_in_nedb(r, checkpoint_seq=cks[-1]["_seq"], db_path=a.ledger)
        print(f"\nledger: eval seq={res['seq']} wrote={res['wrote']}")
    return 0


def cmd_lineage(a) -> int:
    db = _ledger(a.ledger)
    print("=== ledger contents ===")
    for coll in ("datasets", "training_runs", "checkpoints", "evals"):
        rows = db.query(f"FROM {coll}")
        print(f"  {coll:<14} {len(rows)} rows")
    print(f"\nverify(): {db.verify()}")
    print(f"head:     {db.head}")

    print("\n=== TRACE caused_by  (backward: evals -> dataset) ===")
    try:
        for row in db.query("FROM evals TRACE caused_by"):
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            print(f"  {row.get('_id')}: {json.dumps(clean)[:130]}")
    except Exception as e:
        print("  FAILED:", type(e).__name__, e)

    print("\n=== TRACE caused_by REVERSE  (forward: dataset -> evals) ===")
    try:
        for row in db.query("FROM datasets TRACE caused_by REVERSE"):
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            print(f"  {row.get('_id')}: {json.dumps(clean)[:130]}")
    except Exception as e:
        print("  FAILED:", type(e).__name__, e)

    ds = db.query("FROM datasets")
    if ds:
        seq = ds[0]["_seq"]
        print(f"\n=== AS OF {seq}  (replay dataset state) ===")
        for row in db.query(f"FROM datasets AS OF {seq}"):
            print(f"  {row.get('_id')}  train_rows={row.get('train_rows')} "
                  f"seed={row.get('seed')}")
    return 0


def cmd_cast(a) -> int:
    from .inference import Cast
    c = Cast.from_pretrained(a.out)
    print(repr(c), file=sys.stderr)
    for prompt in a.prompt:
        r = c.try_plan(prompt)
        print(f'\n  "{prompt}"')
        print(f"   nql: {r['nql']}")
        if r["ok"]:
            print(f"   plan: {json.dumps({k: v for k, v in r['plan'].items() if v not in (None, [], False)})}")
        else:
            print(f"   INVALID: {r['error']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cast", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/v1")
    p.add_argument("--ledger", default="ledger")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="build + content-address the corpus")
    g.add_argument("--train", type=int, default=200_000)
    g.add_argument("--eval", type=int, default=4_000)
    g.add_argument("--holdout", type=int, default=2_000)
    g.add_argument("--seed", type=int, default=1337)
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("tokenizer", help="fit the vocab")
    t.add_argument("--min-freq", dest="min_freq", type=int, default=2)
    t.add_argument("--max-vocab", dest="max_vocab", type=int, default=4096)
    t.set_defaults(fn=cmd_tokenizer)

    r = sub.add_parser("train", help="train (resumable)")
    r.add_argument("--steps", type=int, default=14_000)
    r.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    r.add_argument("--lr", type=float, default=3e-3)
    r.add_argument("--warmup", type=int, default=400)
    r.add_argument("--eval-every", dest="eval_every", type=int, default=500)
    r.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=500)
    r.add_argument("--n-layer", dest="n_layer", type=int, default=4)
    r.add_argument("--n-embd", dest="n_embd", type=int, default=256)
    r.add_argument("--n-head", dest="n_head", type=int, default=4)
    r.add_argument("--block-size", dest="block_size", type=int, default=128)
    r.add_argument("--dropout", type=float, default=0.05)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--seed", type=int, default=1337)
    r.add_argument("--no-resume", dest="no_resume", action="store_true")
    r.add_argument("--max-seconds", dest="max_seconds", type=float, default=None)
    r.add_argument("--run-id", dest="run_id", default=None)
    r.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="score with the real parser")
    e.add_argument("--split", default="eval", choices=["eval", "holdout", "train"])
    e.add_argument("--data-path", dest="data_path", default=None)
    e.add_argument("--limit", type=int, default=1000)
    e.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    e.add_argument("--show-failures", dest="show_failures", type=int, default=0)
    e.set_defaults(fn=cmd_eval)

    l = sub.add_parser("lineage", help="TRACE the ledger")
    l.set_defaults(fn=cmd_lineage)

    c = sub.add_parser("cast", help="cast prompts to NQL")
    c.add_argument("prompt", nargs="+")
    c.set_defaults(fn=cmd_cast)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
