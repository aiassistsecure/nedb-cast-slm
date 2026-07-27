# PROVENANCE

How `nedb-cast-slm` records its own lineage, and the queries that read it back.

The claim is narrow and testable: **given any checkpoint, you can recover the exact
data that trained it and the exact scores it earned, and prove the record wasn't
edited.** Not from a spreadsheet or a naming convention — from the database, with
a query.

---

## The ledger

Four collections in one NEDB database, chained by `caused_by`:

```
datasets ──▶ training_runs ──▶ checkpoints ──▶ evals
```

Each arrow is a real causal edge in the engine's DAG, not a foreign key by
convention. `caused_by` takes a list of **integer sequence numbers** — the seq of
the op that wrote the causing document.

> This is the API detail that cost the most time to learn. `caused_by` is *not*
> hashes and *not* document dicts. Passing a dict yields
> `'<' not supported between instances of 'str' and 'int'`, which reads like an
> engine bug and is not one. See [LORE.md](LORE.md) §I.

```python
ds  = db.put("datasets", ds_id, {...})
run = db.put("training_runs", run_id, {...}, caused_by=[ds["_seq"]])
ck  = db.put("checkpoints", ck_id,  {...}, caused_by=[run["_seq"]])
ev  = db.put("evals", ev_id,        {...}, caused_by=[ck["_seq"]])
```

### `datasets`

The corpus manifest. Never the corpus itself — the rows live in JSONL; the ledger
records their **identity**.

| field | meaning |
|---|---|
| `_id` | `ds_<blake2b>` — content address (see below) |
| `source_fingerprint` | hash of `grammar.py` + `sampler.py` + `paraphrase.py` |
| `seed` | RNG seed |
| `unique_plans` | distinct plans generated |
| `train_rows` / `eval_rows` / `holdout_rows` | split sizes |
| `train_plans` / `eval_plans` / `holdout_plans` | split sizes *by plan* |
| `clause_p` | the clause-probability config, serialized |
| `domains` | schema domains included |

### `training_runs`

| field | meaning |
|---|---|
| `dataset_id` | human-readable pointer (the causal edge is the real link) |
| `params_m`, `n_layer`, `n_embd`, `n_head`, `block_size` | architecture |
| `steps_planned`, `batch_size`, `lr`, `dropout`, `seed` | hyperparameters |
| `device`, `threads`, `host` | where it ran |
| `tok_per_s_observed` | measured throughput |

### `checkpoints`

| field | meaning |
|---|---|
| `step` | training step |
| `train_loss`, `eval_loss` | loss at that step |
| `path` | where the `.pt` lives |

### `evals`

| field | meaning |
|---|---|
| `step` | which checkpoint step was scored |
| `split` | `eval` or `holdout` |
| `n` | examples scored |
| `valid_rate` | fraction that parse at all |
| `exact_match` | fraction canonically equal to gold |
| `per_clause` | per-clause accuracy, serialized |

---

## Content addressing

A dataset's `_id` is a BLAKE2b digest over three things:

1. **the generator source** — `grammar.py`, `sampler.py`, `paraphrase.py`
2. **the config** — row counts and seed
3. **the output** — the actual JSONL bytes

```python
h = hashlib.blake2b(digest_size=16)
h.update(source_fingerprint.encode())
h.update(json.dumps(config, sort_keys=True).encode())
for split in ("train", "eval", "holdout"):
    h.update(hashlib.blake2b(open(paths[split], "rb").read(), digest_size=16).digest())
run_id = "ds_" + h.hexdigest()
```

Consequences, both intended:

**Same seed + unchanged code ⇒ same id ⇒ no write.** Re-running the generator is a
no-op. Verified across three separate OS processes:

```
run 1: ds_16d42c85281b3182294a38c27295b398   wrote: true
run 2: ds_16d42c85281b3182294a38c27295b398   wrote: false
run 3: ds_16d42c85281b3182294a38c27295b398   wrote: false
```

**Change one word of the paraphraser ⇒ new id.** It is a different dataset now and
must never be mistaken for the old one, even at identical row counts.

Getting this right required removing a `set()` from the split logic — see
[LORE.md](LORE.md) §II. `list(set(strings))` iterates in `PYTHONHASHSEED`-dependent
order, which silently produced a fresh id on every run while appearing to work.

---

## The queries

### Backward — where did this score come from?

```python
db.query("FROM evals TRACE caused_by")
```

```
ck_1     {"step": 20000, "loss": 0.21}
run_1    {"params": "3.32M", "steps": 14000}
ds_v1    {"name": "nql-corpus-v1", "examples": 200000, "seed": 1337}
```

Three hops, one query. An eval score to the checkpoint, the run, and the corpus —
without a single join written by hand.

### Forward — what came out of this dataset?

```python
db.query("FROM datasets TRACE caused_by REVERSE")
```

```
run_1    {"params": "3.32M", "steps": 14000}
ck_1     {"step": 20000, "loss": 0.21}
ev_1     {"plan_exact_match": 0.87, "held_out": 0.79}
```

Useful in the direction that actually hurts: *a bug was just found in the
paraphraser — which models are contaminated?* Ask the dataset.

### Replay — what exactly did it train on?

```python
db.query(f"FROM datasets AS OF {seq}")
```

The corpus manifest as it existed at that sequence, including counts and config,
even after newer datasets were written.

### Integrity — has the record been edited?

```python
db.verify()   # True
db.head       # b677a69d2542e061b883f5c8ff79fe7b7a98e96892cee68a6ae03535dcb1e7dc
```

The log is BLAKE2b hash-chained. An eval score cannot be quietly improved after
publication — altering any historical row breaks the chain and `verify()` returns
False. Publishing `head` alongside a result pins the entire ledger at that moment.

---

## Why bother

Standard ML provenance is a directory name and a hope:

```
checkpoints/run_v3_final_v2_ACTUALLY_final/
```

Which data? Which paraphraser version? Was the eval run before or after that
tokenizer fix? Usually unknowable within a week, and unknowable *with certainty*
immediately.

Here those questions are queries with exact answers, because the substrate has
causal edges, time travel, and tamper-evidence as native primitives. This project
did not build a provenance system — it used a database that already had one.

Which is the honest reason `nedb-cast-slm` exists. A 3.3M-parameter model that
writes NQL is a neat trick. A 3.3M-parameter model that can **prove which data
made it** is a different category of object, and it's only cheap to build when the
engine underneath already thinks in causality.

---

## Reproducing the chain

```bash
python -m cast.cli generate --train 200000 --seed 1337   # writes datasets row
python -m cast.cli train --steps 14000                   # writes training_runs + checkpoints
python -m cast.cli eval --split eval                     # writes evals row
python -m cast.cli eval --split holdout                  # writes evals row
python -m cast.cli lineage                               # TRACE it back
```

Run `generate` twice — the second is a no-op, and the ledger stays at one dataset
row. That's the point.
