# nedb-cast-slm

**A 3.3M-parameter language model that turns a sentence into a database query — and can prove where it came from.**

```python
from cast import Cast

caster = Cast.from_pretrained("runs/v1")

caster.cast("top 5 stylists in winter park")
# FROM stylists WHERE city = "winter park" LIMIT 5

caster.cast("what caused these checkpoints")
# FROM checkpoints TRACE caused_by
```

No GPU. No API key. No network call. It trained from scratch on **two CPU cores**
in about half an hour, and it fits in a file smaller than most JavaScript bundles.

---

## Why this exists

Every "natural language to query" feature you've used is a round-trip to someone
else's datacenter. You type a sentence, it goes to a frontier model, a query comes
back, you pay per token, and you wait. For a query language with **ten clauses and
six operators**, that is an absurd amount of machinery.

NQL — the [NEDB](https://github.com/Eth-Interchained/nedb) query language — is small
enough to learn completely. So we taught a small model to do it, and left the
frontier models for problems that actually need them.

The interesting part is not the model. It's what the model is **attached** to.

---

## The two-way attachment

`nedb-cast-slm` doesn't merely target NEDB. NEDB is on both ends of it.

### Direction 1 — the engine writes its own training data

NEDB ships a parser: `nedb/query.py`. It takes NQL text and returns a plan dict.

That parser is doing three jobs here that would otherwise cost money and time:

**It is the data generator.** Sample a random plan, render it as NQL, render a
human paraphrase. Perfect labels, infinite supply, zero annotation cost. We built
200,000 examples in **16.5 seconds**.

**It is the grader.** We don't score string equality — we score whether the
*parsed plan* matches. `FROM orders WHERE total > 99` and
`from orders where total>99` are the same query, and the model gets full credit
for both. Anything the engine can't parse fails deterministically.

**It is the gate.** Not one example enters the corpus unless it round-trips
through the real parser back to a canonically identical plan. A generator bug
crashes the build instead of quietly teaching the model invalid syntax.

> Most text-to-DSL projects hand-write a verifier and then hope it's right.
> We didn't write one. It already shipped, and it's the same code the database
> uses in production.

### Direction 2 — the model records its own lineage

Every artifact of training is a document in NEDB, chained by `caused_by`:

```
datasets  ──caused_by──▶  training_runs  ──caused_by──▶  checkpoints  ──caused_by──▶  evals
```

Which buys three things you cannot get from a folder of `.pt` files:

```python
db.query("FROM evals TRACE caused_by")
# → the checkpoint, the run, and the exact dataset behind this score
```

**`TRACE caused_by`** — full lineage of any checkpoint. Which data trained it,
which run produced it, which eval graded it. One query.

**`AS OF <seq>`** — replay the exact corpus any checkpoint was trained on.
Reproducibility as a query instead of a spreadsheet.

**`verify()`** — the ledger is hash-chained and tamper-evident. Nobody edits an
eval score after the fact.

The dataset id is a BLAKE2b hash over the generator source, the config, and the
resulting pairs. Same seed and unchanged code produce the same id, so re-running
writes **nothing** — verified across three separate processes. Change one line of
the paraphraser and the id changes, because it is now a different dataset and
should never be confused for the old one.

Models that show their work. That's the whole idea.

---

## Results

Trained on 2 vCPU (Xeon @ 2.9GHz), 4GB RAM, no GPU. Metrics are **plan-dict
exact match** against the real parser, on plans held out from training.

| | |
|---|---|
| Parameters | 3.34M |
| Vocabulary | 581 tokens |
| Training | 14,000 steps in **40.9 min** on 2 vCPU |
| Throughput | 7,081 tok/s (CPU) |
| Final eval loss | 0.0099 |
| **Valid NQL rate** | **100.0%** (eval) · 98.8% (holdout) |
| **Exact plan match** | **92.3%** (eval) · **63.1%** (holdout) |

Two numbers, both reported, because they measure different things. **Eval** uses phrasings
drawn from the same distribution as training — that's the competence number. **Holdout** uses
structures the model has never seen (inverted clause order, question framing, appositive
asides) built from in-vocabulary words — that's the compositional-generalisation number, and
it is deliberately adversarial.

Accuracy by clause, so a weak clause can't hide behind a good average:

| clause | eval | holdout |
|---|---|---|
| `TRACE ... REVERSE` | 97.7% | 71.0% |
| `TRACE` | 96.5% | 68.1% |
| `TRAVERSE` | 93.3% | 61.4% |
| `VALID AS OF` | 92.6% | 58.4% |
| `WHERE` (single) | 91.2% | 56.6% |
| `LIMIT` | 91.1% | 60.2% |
| `SEARCH` | 90.5% | 58.5% |
| `AS OF` | 88.3% | 69.4% |
| `ORDER BY` | 87.7% | 51.1% |
| `WHERE` (multi-predicate) | 85.1% | 61.2% |
| `GROUP BY` | 80.2% | 52.9% |
| aggregate | 77.0% | 55.0% |

### What it gets wrong

Being specific, because "72.8%" without a failure mode is marketing:

**Duplicated predicates on comma-appositive structures.** The clearest holdout failure:

```
"list products, price cheaper than 182, aggregated by category with sum of price"
  pred: WHERE price < 182.0 AND price < 182.0 GROUP BY category SUM price   ← duplicated
  gold: WHERE price < 182.0 GROUP BY category SUM price
```

It gets the semantics right and emits the predicate twice — it reads the closing comma as a
conjunction, because that construction appears only in holdout. This is a real compositional
limit, not a measurement artifact.

**Long digit runs.** `"blocks above height 400000"` → `WHERE height > 4000`. Numbers are
tokenized digit-by-digit, so copying a 6-digit literal means six sequential correct
predictions with no positional anchor.

**Clause attachment under reordering.** When a sort clause is interposed between the subject
and its conditions, fields occasionally attach to the wrong clause.

It is honest about the boundary: **this model interprets short prompts into a
constrained grammar. It does not write code, and nothing at 3.3M parameters will.**

---

## Install

```bash
pip install nedb-cast-slm
```

## Use

```python
from cast import Cast
from nedb import NEDB

caster = Cast.from_pretrained("runs/v1")
db = NEDB("./mydb")

# NQL text
caster.cast("invoices that are overdue limit 10")
# 'FROM invoices WHERE status = "overdue" LIMIT 10'

# parsed plan — raises if the model emitted something invalid
caster.plan("memories with importance 5 grouped by category count")
# {'from': 'memories', 'where': [['importance','=',5]], 'group_by': 'category', ...}

# non-raising variant, for UIs that want to show the failure
caster.try_plan("...")
# {'ok': True, 'nql': '...', 'plan': {...}}

# cast and execute in one step
caster.run("paid orders over $99", db)
```

`plan()` raising on invalid output is intentional. A query planner that silently
returns *almost* the right query is worse than one that admits it failed.

## Train your own

```bash
python -m cast.cli generate --train 200000     # build + content-address the corpus
python -m cast.cli tokenizer                   # fit the vocab
python -m cast.cli train --steps 14000         # train (resumable)
python -m cast.cli eval --split holdout        # score with the real parser
python -m cast.cli lineage                     # TRACE the ledger
```

Every step writes to the NEDB ledger. `train` checkpoints continuously and resumes
exactly where it stopped — built that way because this was developed in an
ephemeral container that could be reclaimed mid-run.

---

## How it's built

```
cast/grammar.py     six synthetic domains (shop, salon, chain, agent, crm, ops)
cast/sampler.py     random valid plan → canonical NQL
cast/paraphrase.py  plan → varied human phrasing
cast/dataset.py     corpus build, content addressing, NEDB ledger
cast/tokenizer.py   516-token word-level vocab, digit-split numbers
cast/model.py       ~3.3M param causal transformer (d=256, L=4, H=4, T=128)
cast/train.py       prompt-masked loss, bucketed batching, resumable
cast/evaluate.py    plan-equality grading, per-clause breakdown
cast/inference.py   the public Cast API
```

**Six domains, not one.** A model trained only on `orders.total` is useless in a
studio where every user has their own collections. Training across six unrelated
schemas forces it to learn query *shape* rather than memorize field names.

**Prompt-masked loss.** Loss is computed only on tokens after `<sep>`. Training on
the prompt too would spend scarce capacity learning to model English, which is not
the job.

**Bucketed batching.** Sequences average 39 tokens but max at 125. Padding
everything to 128 wasted ~60% of compute; bucketing by length made training
**2.4x faster** than the original estimate.

**516 tokens.** A tiny vocab keeps the embedding table cheap, leaving the
parameter budget for layers that actually reason. UNK rate is 0.0000%.

---

## The Rust core is real

Inference also runs as a **zero-dependency Rust crate** (`rust/nedb-cast-core`) — no Python,
no PyTorch. Verified in CI against the PyTorch reference on every push:

```
parity: 20 prompts, max |Δlogit| = 7.629e-6  (tol 1.0e-4)
decode parity: 20 / 20 exact
test result: ok. 5 passed (unit) · 4 passed (parity) · 1 passed (doctest)
```

Max logit deviation **7.6e-06**, and all 20 decoded NQL strings byte-identical. That's what
makes crates.io / npm / PyPI distribution legitimate rather than a Python shim in a trenchcoat.
See [rust/README.md](rust/README.md).

## Read next

- **[docs/LORE.md](docs/LORE.md)** — the bugs. A 55-point accuracy swing that was
  never the model's fault, and the "obvious" diagnosis that was dead wrong twice.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — why each design choice, and
  what we rejected.
- **[docs/PROVENANCE.md](docs/PROVENANCE.md)** — the ledger schema and the lineage
  queries in full.

---

## Credits

Built by **[Interchained LLC](https://interchained.org)** on
**[NEDB](https://github.com/Eth-Interchained/nedb)** — the versioned, bi-temporal,
causally-provable embedded database that makes the provenance half of this project
possible.

Developed on HyperAgent. Interchained is a Founding 500 member of HyperAgent and
received 20,000 HyperAgent inference credits.

*Lightning strikes, thunder roars, code appears.*

## License

BUSL-1.1
