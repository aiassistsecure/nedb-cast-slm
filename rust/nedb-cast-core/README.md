# nedb-cast-slm (Rust core)

Pure-Rust inference for [**nedb-cast-slm**](https://github.com/aiassistsecure/nedb-cast-slm) —
a 3.34M-parameter model that casts short English prompts into NEDB query plans (NQL).

**No Python. No PyTorch. No dependencies at all** — the `[dependencies]` table is empty.

```rust
use nedb_cast_slm::Cast;

let cast = Cast::load("model.cast")?;
let nql  = cast.cast("top 5 stylists in winter park");
// FROM stylists WHERE city = "winter park" LIMIT 5
```

## Why

Training needs PyTorch. Inference does not — it is a few thousand f32 matmuls over a tiny
model. Splitting them means the *usable* half has no runtime dependency on Python, which is
what makes real crates.io distribution possible instead of a shim that shells out to an
interpreter it cannot guarantee exists.

## Numerical parity with PyTorch

Enforced in CI on every push, three levels deep:

| level | assertion |
|---|---|
| tokenization | Rust ids **exactly** equal Python's |
| logits | every final-position logit within `1e-4` |
| output | generated ids **and** decoded NQL match **exactly** |

Measured on the released model:

```
parity: 20 prompts, max |Δlogit| = 7.629e-6  (tol 1.0e-4)
decode parity: 20 / 20 exact
```

A port that is merely *close* is wrong: greedy decoding takes an argmax, so ~1e-3 of drift
flips any near-tie and silently emits a different query.

### The GELU trap

PyTorch's default `F.gelu` is the **exact erf** formulation, not the tanh approximation:

```
erf-based  (used here): max abs err 9.537e-07
tanh approximation:     max abs err 4.735e-04   ← 497x worse, at the tolerance limit
```

If you touch `gelu()` in `src/model.rs`, re-run the comparison.

## Weights

Weights ship as GitHub **release assets**, not in the crate. Download `model.cast` from
[the releases page](https://github.com/aiassistsecure/nedb-cast-slm/releases) and pass its
path to `Cast::load`. It is the same container the Python package consumes, so both
languages load byte-identical weights.

## Docs

Full format spec, architecture notes, and the development lore (four bugs, none of which
were in the model) live in the [repository](https://github.com/aiassistsecure/nedb-cast-slm):
`rust/README.md`, `docs/ARCHITECTURE.md`, `docs/LORE.md`.

## License

BUSL-1.1 · [Interchained LLC](https://interchained.org)
