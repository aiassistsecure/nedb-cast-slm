# LORE

Every number in the README was wrong at least once. This is the record of how,
because the failures taught more than the successes did.

---

## I. The parser we didn't have to write

The project began with a wrong belief.

Carried into the session was a note from an earlier build: *"`TRACE caused_by` is
broken upstream — throws `'<' not supported between instances of 'str' and 'int'`
on nedb 2.6.1."* The whole provenance story depended on `TRACE` working, so it was
task one: reproduce it on 2.7.2, then plan around the breakage.

It reproduced immediately. Same error, same line — `engine.py:738`:

```python
if cause_seq < len(self.log.ops):
```

Confirmation. A real upstream bug, still unfixed a version later.

Except the next step was to read the surrounding code rather than trust the
conclusion. Line 739:

```python
op = self.log.ops[cause_seq]
```

It **indexes the op log directly**. `caused_by` is not a list of hashes. It is a
list of *integer sequence numbers*. The test had been written passing whole
document dicts:

```python
run = db.put("training_runs", "run_1", {...}, caused_by=[ds])   # ds is a dict
```

`str < int` blew up exactly as any sane implementation would. Rewritten with seqs:

```python
run = db.put("training_runs", "run_1", {...}, caused_by=[ds["_seq"]])
```

Four-hop lineage, both directions, first try:

```
FROM evals TRACE caused_by            → ck_1 → run_1 → ds_v1
FROM datasets TRACE caused_by REVERSE → run_1 → ck_1 → ev_1
verify()                              → True
```

**There was never an upstream bug.** There was a caller passing the wrong type,
and a note that had hardened into folklore. It had been repeated confidently
across sessions without once being tested.

> **Lesson.** A reproduced error is not a diagnosed error. The stack trace was
> real; the conclusion drawn from it was invented. Read the line *after* the one
> that threw.

A second, smaller one fell out of the same test: `db.head` is a property, not a
method. `db.head()` raises `'str' object is not callable`. Harmless — unless it's
inside a training loop that runs for half an hour before reaching it.

---

## II. The dataset that was never the same twice

The content-addressing pitch is clean: hash the generator source, the config, and
the output; identical inputs produce an identical id; re-running writes nothing.

It was tested by generating twice with the same seed.

```
run 1: ds_2f3c0b7ef86200a7860aa945d5689bfe
run 2: ds_f471dea5efef64ddfd21ff4bdd8c248f
```

Two ids. `wrote: true` both times — two ledger rows for what should be one
dataset. The plan sampler was verified deterministic (same seed, identical plans).
The source fingerprint was stable across calls. The data files were byte-identical
in size.

The culprit was one word:

```python
eval_keys = set(keys[:n_eval_plans])       # ← set
...
kl = list(key_list)                        # ← back to list, in WHOSE order?
```

`list(set_of_strings)` iterates in an order determined by string hashes, and
Python randomizes the string hash seed **per process**. Rows were written in a
different order every run, so the content hash differed every run — while every
individual row was identical.

Fix: never round-trip through a set when order is load-bearing.

```python
eval_keys = keys[:n_eval_plans]            # stays a list
```

Verified across three separate processes:

```
run 1: ds_16d42c85281b3182294a38c27295b398   wrote: true
run 2: ds_16d42c85281b3182294a38c27295b398   wrote: false
run 3: ds_16d42c85281b3182294a38c27295b398   wrote: false
```

> **Lesson.** This one is nastier than a crash. The feature *looked* like it
> worked — ids were generated, rows were written, nothing errored. Only a test
> that asserted the actual invariant ("same seed ⇒ same id") could see it. A
> provenance system that silently mints a new identity for unchanged data is
> worse than no provenance system, because you'd trust it.

---

## III. The 76% ceiling hiding in the detokenizer

With the corpus built, the tokenizer was fitted: 516 tokens, 0.0000% UNK, mean
sequence 39 tokens. Clean.

Then the round-trip test — encode NQL, decode it, does it still parse to the same
plan?

**76.05%.**

Nearly a quarter of the corpus could not survive its own tokenizer. And crucially:
**this was a hard ceiling on final accuracy.** No amount of training could exceed
it, because the model's targets were being corrupted before it ever saw them.

Three bugs, all in `_detok`:

```
orig: FROM products VALID AS OF "2026-10-18" ORDER BY title ASC
back: FROM products VALID AS OF "2026 -10 -18" ORDER BY title ASC
```

Dates lost their hyphens. The digit-merge loop accepted `.` as a continuation
character but not `-`, so `2026-10-18` decoded as three separate literals.

```
orig: FROM events SEARCH "rate limit" TRACE caused_by REVERSE
back: FROM events SEARCH "rate LIMIT" TRACE caused_by REVERSE
```

`limit` is an NQL keyword, so it was uppercased — **including inside a quoted
string literal**, corrupting user data. Keyword casing has to be quote-aware.

And `!=` lost its leading space, which the parser tolerates but which broke
byte-level comparison.

After fixing all three: **20,000/20,000, 100.00%.**

> **Lesson.** Had this shipped, training would have plateaued near 76% and the
> obvious response would have been to blame the model — add layers, add data,
> add steps. Hours of compute chasing a bug in forty lines of string handling.
> **Test the boring parts.** The tokenizer is not where the intelligence lives,
> which is exactly why nobody looks at it.

---

## IV. The 55-point bug that was never the model

This is the one worth the whole document.

Training was healthy — loss falling smoothly, eval below train, no divergence:

| step | train | eval |
|---|---|---|
| 500 | 1.6436 | 0.7319 |
| 1500 | 0.3771 | 0.2179 |
| 2500 | 0.1871 | 0.1153 |

But exact-match was catastrophic. **1.9% at step 1500. 11.9% at step 2500.**

A loss of 0.11 with 11.9% exact-match is a contradiction. Low loss means most
tokens are right; near-zero exact-match means nearly every query has at least one
token wrong. Both can be true — exact match is unforgiving — but not *this* far
apart.

So the failures were categorized. A clear signal emerged:

```
  58  wrong_field
  57  dropped predicate (pred 0 vs gold 1)
  31  wrong_op
  20  wrong_value
```

And the samples were damning:

```
gold: WHERE balance < 2942.0 AND label = "governance"
pred: WHERE watch_only = TRUE AND watch_only = TRUE      ← repeated predicate
gold: WHERE sender = "marisa" AND amount < 4299.1
pred: WHERE sender = "marisa" AND sender = "marisa"      ← copied predicate 1 twice
```

A textbook small-model failure: it cannot track which fields it has already
emitted, so it loops. The diagnosis wrote itself — **3.3M parameters is too small
to maintain state across a multi-predicate clause.** The fix wrote itself too:
copy-attention, or a pointer mechanism, or simply a bigger model.

It was a good story. Coherent, well-supported, and matched a known failure mode in
the literature.

One check first — accuracy as a function of query complexity. If the model is
capacity-bound, accuracy must **fall** as clauses accumulate:

```
exact-match by clause count:
  1 clause :   0.9%
  2 clauses:   4.9%
  3 clauses:   7.7%
  4 clauses:  12.5%   ← rising
```

Backwards. Accuracy *increased* with complexity. And 1-clause queries at **0.9%**
is absurd — `FROM services LIMIT 5` is the easiest output in the corpus. No
capacity limit produces that shape.

Something was wrong with the *measurement*. Same checkpoint, same prompts, batched
vs. one-at-a-time:

```
                                          batched                     single
FROM products TRAVERSE reviewed LIMIT 10   … GROUP BY category SUM stock ✗   exact ✓
FROM appointments SEARCH "no show" …       SEARCH "consultation"          ✗   exact ✓
FROM customers AS OF 666 LIMIT 5           … GROUP BY occurred_at AVG …   ✗   close ✓
```

The evaluator was the bug.

`predict_batch` left-padded prompts to a common length. But the model uses
**learned positional embeddings and no padding mask**. Pad tokens shift every real
token's position *and* get attended to. A short prompt batched with long ones is
read at the wrong offsets through noise.

Which explains the impossible curve exactly: **longer prompts need less padding,
so they were less corrupted.** The "capacity limit" was a padding gradient wearing
a convincing costume.

Fix — bucket by identical encoded length so no batch ever contains padding:

```python
buckets.setdefault(len(enc[i]), []).append(i)
```

Verified: batched output now matches single-sequence output on 60/60 prompts.

Re-scored, same checkpoint, nothing retrained:

| | before | after |
|---|---|---|
| valid NQL | 71.6% | **98.8%** |
| exact match | 11.9% | **66.6%** |

**Fifty-five points.** The model had been performing well the entire time. The
ruler was bent.

> **Lesson, and it's the important one.** The wrong diagnosis was not careless —
> it was *well-evidenced*. Failure counts supported it. Sample outputs supported
> it. It matched a real documented phenomenon. Every piece of evidence pointed at
> the model, and every piece was a downstream symptom of a bug three files away.
>
> The thing that broke it open was a number that was **impossible**, not merely
> bad. 1-clause queries at 0.9% could not be true. Accuracy rising with complexity
> could not be true. A believable-but-wrong story survives scrutiny; an impossible
> number does not.
>
> **When the evidence fits the story a little too well, go find the number that
> cannot possibly be right.** Then explain that one.

---

## V. What remains genuinely hard

With the measurement fixed, the real weakness stands clear — and it is the one
predicted before training started, just at a far higher baseline:

**Long digit runs.**

```
"blocks above height 400000"  →  WHERE height > 4000
```

Numbers are tokenized digit-by-digit, so copying a 6-digit literal requires six
sequential correct predictions with no positional anchor. Every other approach has
a worse tradeoff at this vocabulary size, so this is the accepted cost — and the
first thing a v2 should address, probably with a copy mechanism over prompt
tokens rather than free generation.

That distinction matters: this is a **real** limitation, arrived at after three
fake ones were cleared away.

---

## Postscript: the pattern

Four bugs. Not one lived in the model.

| bug | lived in | looked like |
|---|---|---|
| `TRACE` "upstream bug" | the caller | a database bug |
| non-deterministic dataset id | `list(set(...))` | working correctly |
| 76% tokenizer ceiling | 40 lines of string handling | a model that plateaued |
| 55-point accuracy loss | the evaluator's padding | a capacity limit |

The model was the only component that never misbehaved. Everything around it did.

That's not a coincidence — it's where attention goes. The architecture is the
interesting part, so it gets read carefully and reviewed twice. The tokenizer, the
batcher, the data loader, the eval harness are plumbing, so they get written once
and trusted forever.

**In a machine learning project, the plumbing is where the bugs live, and they
disguise themselves as model failures.**

The round-trip gate is the durable defense. Every generated example must parse
through the real engine parser back to a canonically identical plan, or the build
fails loudly. It cost about twenty lines. It is the reason a generator bug can
never quietly poison the corpus — and the reason the remaining 27% of errors are
honestly the model's, and not something else wearing its clothes.

*3 > 1.*
