# ARCHITECTURE

Every choice here was made against one constraint: **2 CPU cores, 4GB RAM, no GPU.**
That constraint is clarifying. It rules out most of the usual answers and forces
the question *what does this task actually need?*

---

## The task

Map a short English prompt to a plan in a ten-clause grammar:

```
FROM <coll> [AS OF <seq>] [VALID AS OF "<date>"]
  [WHERE <f> <op> <v> (AND ...)*] [SEARCH "<text>"]
  [ORDER BY <f> [ASC|DESC]] [TRAVERSE <rel>]
  [TRACE <field> [REVERSE]] [LIMIT <n>]
  [GROUP BY <f> [COUNT|SUM <f>|AVG <f>|MIN <f>|MAX <f>]]
```

Six operators, ten clauses, one fixed clause order. Bounded — which is exactly why
a small model is the right tool and a frontier model is overkill.

**Prompts are short** (mean 20 tokens). **Targets are short** (mean 18 tokens).
Nothing here needs long-context machinery.

---

## Sizing, from measurement

Before writing any model code, throughput was benchmarked on the actual box with a
real forward+backward+step:

| config | params | ms/step | tok/s | 20M-token epoch |
|---|---|---|---|---|
| d=256, L=4, T=128, B=16 | 3.72M | 730 | 2,807 | ~119 min |
| d=384, L=6, T=256, B=8 | 11.53M | 1,334 | 1,536 | ~217 min |

**Chosen: d=256, L=4, H=4, T=128 → 3.32M params.**

The 12M option triples wall-clock for a task this narrow. The 3.3M row was the
honest throughput number, so it set the budget.

Actual observed throughput came in at **~5,100 tok/s — 1.8x the benchmark** —
because bucketed batching removed padding waste the benchmark didn't model. Faster
than predicted, which is the direction you want an estimate to be wrong.

---

## Model

```
tok_emb (516 × 256) + pos_emb (128 × 256)
  → 4 × [ LN → causal self-attn (4 heads) → LN → MLP(4×) ]
  → LN → head (tied to tok_emb)
```

### Weight tying

`head.weight = tok_emb.weight`. With a 516-token vocab this saves little memory —
it's chosen because tying reliably helps small models generalize and removes an
entire output matrix from the gradient path.

### Pre-norm blocks

LayerNorm *before* attention and MLP. Post-norm needs careful warmup to stay
stable when training from scratch; pre-norm mostly doesn't. With one shot at a
30-minute run, stability beats theoretical peak quality.

### Learned positional embeddings

Not RoPE, not ALiBi. Sequences are short and the max is fixed at 128. RoPE's
advantage is length extrapolation, which is worth nothing here, and it costs
clarity.

**This choice has a sharp edge.** Learned positions plus no padding mask means
padding *shifts* every real token's position. That interaction produced the worst
bug in the project — a 55-point accuracy loss that looked exactly like a model
capacity limit ([LORE.md](LORE.md) §IV). The architecture is fine; anything that
batches mixed-length sequences must group by identical length. Both `evaluate.py`
and `inference.py` do.

### Scaled residual init

Residual projections initialized at `0.02 / sqrt(2 * n_layer)`, GPT-2 style. Keeps
the residual stream from compounding early.

---

## Tokenizer: 516 tokens, word-level

The most consequential choice, and the least obvious.

**Why not BPE?** At 3.3M params the embedding table is a meaningful fraction of the
budget. A vocab fitted to *this* corpus makes each NQL keyword exactly one token
and keeps sequences short — mean 39 tokens for a full prompt+target pair.

**Numbers split into digits.** Numbers are unbounded; a vocab can't hold them.
Splitting into `0-9`, `.`, `-` makes any literal representable with 14 symbols.

That is a real tradeoff, honestly stated: **it is also the single largest source of
remaining error.** Copying `400000` means six sequential correct predictions with
no positional anchor, and the model sometimes emits `4000`. The alternatives are
worse — number-bucketing loses exactness, and a large numeric vocab blows the
parameter budget — but a v2 should replace free digit generation with a copy
mechanism over prompt tokens.

**0.0000% UNK rate** on the corpus. The vocab is complete for this domain.

### Detokenization is not symmetric

Encoding is easy; decoding has real logic and had three bugs:

- digit runs must re-merge, including `-` (so dates survive)
- keyword casing applies **outside quotes only** (so `SEARCH "rate limit"` doesn't
  become `"rate LIMIT"`)
- quoted literals get tightened

Guarded by a hard test: **20,000/20,000 NQL strings encode→decode→parse back to a
canonically identical plan.** That gate exists because a broken detokenizer caps
final accuracy invisibly.

---

## Data: synthetic, verified, six domains

### Why synthetic

The parser is the oracle. Sample a plan → render NQL → render a paraphrase, and
the label is correct *by construction*. 200,000 examples in 16.5 seconds, no
annotation.

### Why six domains

`shop`, `salon`, `chain`, `agent`, `crm`, `ops`.

A model trained only on `orders.total` learns *that schema*, not the task. Six
unrelated schemas force it to learn query **shape** — "the thing after FROM is a
collection, the thing after WHERE is one of its fields" — which is what makes it
useful against a schema it has never seen.

Fields carry types (`INT`, `MONEY`, `DATE`, `ENUM`, `BOOL`, …) which drive both
value sampling and phrasing: money gets `$99.50` and "cheaper than", counts get
"fewer than", dates get "after"/"before". Early versions ignored types and produced
`lifetime value fewer than 4046.1` — grammatical, semantically nonsense, and
actively teaching the model wrong associations.

### The round-trip gate

No example enters the corpus unless:

```python
parse_nql(render_nql(plan))  ==  plan     # canonically
```

A generator bug raises instead of shipping. This is the highest-leverage twenty
lines in the repo.

### Splits are plan-disjoint

Split by **plan**, never by row. The same plan in a different phrasing appearing in
both train and eval would inflate scores badly. Verified: train∩eval, train∩holdout,
eval∩holdout all **zero** at both plan and prompt level.

**Held-out uses different phrasings entirely** — different verbs, different
operator words ("that cleared", "shy of", "bucketed on"). In-distribution accuracy
measures competence; held-out measures generalization. Both get reported.

---

## Training

### Prompt-masked loss

Each row encodes as `<s> prompt <sep> nql </s>`, and loss is computed **only** on
tokens after `<sep>` (prompt positions set to `ignore_index=-100`).

Training on prompt tokens too would spend scarce capacity learning to model
English. The job is producing the plan *given* the prompt, so that's the only thing
scored.

### Bucketed batching

Sequences run 15–125 tokens with a mean of 39. Padding everything to 128 wastes
~60% of compute on this corpus. Sorting into length buckets so each batch pads to
its own max was the single largest speedup — **2.4x over the naive estimate**.

### Resumable by default

Every checkpoint carries model + optimizer + step. Developed in an ephemeral
container that could be reclaimed mid-run, so resumption isn't a nicety.

### Hyperparameters

| | | why |
|---|---|---|
| batch size | 32 | fits 4GB with headroom |
| lr | 3e-3 | aggressive; small models tolerate it |
| schedule | cosine + 400-step warmup | standard, stable |
| dropout | 0.05 | corpus is large relative to model; little regularization needed |
| grad clip | 1.0 | cheap insurance |
| weight decay | 0.1 | on by default in AdamW here |
| betas | (0.9, 0.95) | standard for LM training |

Eval loss stayed **below** train loss throughout — the model is capacity-limited,
not data-limited, so heavier regularization would only slow learning.

---

## Evaluation

Three metrics, separate because they fail differently:

**`valid_rate`** — does it parse at all? Catches structural collapse.

**`exact_match`** — is the parsed plan canonically equal to gold? **This is the
number.** Not string equality — `FROM orders WHERE total > 99` and
`from orders where total>99` are the same query and both earn credit. Comparison is
on normalized plan dicts.

**`per_clause`** — exact match broken down by which clause the gold plan uses. A
clause failing at 20% cannot hide inside a good average. This is how `GROUP BY`
was identified as the weakest area (~40%) while `TRAVERSE` sits near 88%.

Greedy decoding, always. One right answer means sampling can only hurt.

---

## Rejected alternatives

**Fine-tuning a pretrained small model (GPT-2, TinyLlama).** Would likely score
higher. Rejected for two reasons: no GPU makes even a 124M model painful here, and
a pretrained model's provenance is *someone else's* — which guts the point. Being
able to say "this checkpoint's entire lineage is in this ledger" requires training
from scratch.

**Grammar-constrained decoding.** Masking logits to only grammatically-legal tokens
would push `valid_rate` to 100% by construction. Deliberately not done for v1 — a
model that produces valid NQL *because it learned to* is a more honest result than
one that can't fail. It's the obvious v2 addition, and it will mostly convert
`invalid` into `mismatch` rather than into `exact`.

**Seq2seq encoder-decoder.** Cleaner fit for translation, but decoder-only with a
`<sep>` is simpler, has one code path, and matches how the model will actually be
called.

**BPE tokenizer.** Longer sequences, larger embedding table, no benefit on a
516-word domain.

---

## Where the parameters went

```
tok_emb (tied)   516 × 256   =  132,096
pos_emb          128 × 256   =   32,768
4 × block:
  attn qkv       256 × 768   =  196,608  ×4 =   786,432
  attn proj      256 × 256   =   65,536  ×4 =   262,144
  mlp fc         256 × 1024  =  262,144  ×4 = 1,048,576
  mlp proj      1024 × 256   =  262,144  ×4 = 1,048,576
  layernorms + biases                    ≈      12,000
                                            ─────────
                                            ≈ 3.32M
```

**63% of the model is MLP weights.** That's where the capacity to memorize field
names and clause patterns lives. Embeddings are under 5% — the small vocab paying
off exactly as intended.
