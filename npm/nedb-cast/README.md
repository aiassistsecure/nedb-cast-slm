# @interchained/cast

**A 3.34M-parameter language model that turns a sentence into a database query.**
Pure Rust core. No Python, no PyTorch, no runtime dependencies.

```js
const { pretrained } = require('@interchained/cast');

const cast = await pretrained();          // fetches + verifies weights once
cast.cast('top 5 stylists in winter park');
// FROM stylists WHERE city = "winter park" LIMIT 5

cast.cast('what caused these checkpoints');
// FROM checkpoints TRACE caused_by
```

Targets [NQL](https://github.com/Eth-Interchained/nedb), the NEDB query language.
Trained from scratch on **two CPU cores in 40.9 minutes** — no GPU, no API key.

## Results

| | eval | holdout |
|---|---|---|
| Valid NQL | **100.0%** | 98.8% |
| Exact plan match | **92.3%** | **63.1%** |

Correctness is graded on **plan-dict equality against the real engine parser**,
never string match. Holdout uses sentence structures the model has never seen and
is deliberately adversarial; both numbers are reported because they measure
different things.

## Same weights as Python and Rust

All three load the identical `model.cast` container, so they agree by construction
rather than by luck. CI enforces it on every push against the PyTorch reference:

```
parity: 20 prompts, max |delta logit| = 7.629e-6  (tol 1.0e-4)
decode parity: 20 / 20 exact
```

A port that is merely *close* is wrong — greedy decoding takes an argmax, so ~1e-3
of drift flips a near-tie and silently emits a different query.

## API

```ts
Cast.load(path: string): Cast          // local model.cast
Cast.fromBuffer(buf: Buffer): Cast     // you fetched it yourself
pretrained(opts?): Promise<Cast>       // download once, cache, verify, load

cast.cast(prompt: string): string
cast.castWithLimit(prompt, maxNewTokens): string
cast.logits(prompt): number[]          // confidence, if you want it
cast.vocabSize / cast.nParams
cast.checksumVerified: boolean         // check this
```

**Check `checksumVerified`.** A corrupt model emits plausible-but-wrong queries,
which is worse than a loud failure.

## Weights

Downloaded from the [GitHub release](https://github.com/aiassistsecure/nedb-cast-slm/releases)
on first use and cached under `$CAST_HOME` (default `~/.cache/nedb-cast-slm/<tag>/`),
verified against the `SHA256SUMS.txt` published in the same release. They are not
bundled in the tarball — 13MB times every platform in the prebuild matrix buys
nothing.

## Honest limits

- **Long digit runs.** `"blocks above height 400000"` becomes `WHERE height > 4000`.
  Numbers are tokenized digit-by-digit, so a 6-digit literal needs six sequential
  correct predictions with no positional anchor.
- **Duplicated predicates on comma-appositive phrasing.** Semantics right, predicate
  emitted twice — it reads a closing comma as a conjunction.
- **The boundary:** this interprets short prompts into a constrained grammar. It does
  not write code, and nothing at 3.3M parameters will.

Full failure analysis, including the eight bugs found building it (**none in the
model**): [docs/LORE.md](https://github.com/aiassistsecure/nedb-cast-slm/blob/main/docs/LORE.md).

## License

BUSL-1.1 · [Interchained LLC](https://interchained.org)
