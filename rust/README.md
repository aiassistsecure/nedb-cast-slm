# nedb-cast-core

Pure-Rust inference for **nedb-cast-slm**. No Python, no PyTorch, **no dependencies at all** —
the `[dependencies]` table is empty and stays that way.

```rust
use nedb_cast_core::Cast;

let cast = Cast::load("model.cast")?;
let nql  = cast.cast("top 5 stylists in winter park");
// FROM stylists WHERE city = "winter park" LIMIT 5
```

## Why this crate exists

Training needs PyTorch. Inference does not — it's a few thousand f32 matmuls over a
3.3M-parameter model. Splitting them means the *usable* half of the project has no
runtime dependency on Python at all, which is what makes legitimate crates.io / npm /
PyPI distribution possible instead of publishing shims that shell out to an interpreter
they can't guarantee exists.

It also makes the model embeddable anywhere NEDB already runs.

## The `model.cast` container

One self-contained file, little-endian throughout:

```
  0        8 bytes   magic = b"CASTMDL1"
  8        4 bytes   u32 header_len
  12       H bytes   header JSON (UTF-8)
  12 + H   ...       f32 blob, concatenated, row-major
```

The header carries the config, the **full vocabulary**, a tensor directory
(`{name, shape, offset, length}` with `offset` relative to the start of the blob), and an
FNV-1a-64 checksum over the blob. FNV was chosen because it's a handful of lines of
identical code in both Python and Rust, so the format stays dependency-free on both
sides. It's an integrity check against truncation, not a cryptographic guarantee.

**Weight tying:** `head.weight` *is* `tok_emb.weight` in PyTorch (shared storage), so it
is exported exactly once and the Rust loader aliases the head to it. There is no
`head.weight` entry in the blob — if you see one, the exporter regressed.

Generate it with:

```sh
python scripts/export_weights.py \
    --ckpt runs/v1/ckpt.pt --tokenizer data/tokenizer.json --out model.cast
```

## Numerical parity — how this port is held honest

A Rust forward pass that is merely *close* to PyTorch is wrong. Greedy decoding takes an
argmax, so ~1e-3 of drift flips any near-tie and silently emits a different query. Parity
is therefore enforced on three levels:

| check | assertion |
|---|---|
| tokenization | Rust `encode_prompt` ids **exactly** equal Python's |
| logits | every logit at the final position within `1e-4` (override via `CAST_PARITY_TOL`) |
| output | generated token ids **and** the decoded NQL string match **exactly** |

Fixtures come from `scripts/dump_parity_fixtures.py`, which exports `model.cast` and the
expected outputs from a **single `torch.load`**. That detail matters: the training
checkpoint is rewritten every 500 steps, so exporting weights and fixtures from two
separate loads could describe two different models and fail parity for a spurious reason.

```sh
python scripts/dump_parity_fixtures.py \
    --ckpt runs/v1/ckpt.pt --tokenizer data/tokenizer.json \
    --model-out rust/nedb-cast-core/tests/model.cast \
    --fixtures-out rust/nedb-cast-core/tests/fixtures.json

cd rust/nedb-cast-core && CAST_REQUIRE_PARITY=1 cargo test --release
```

Tests **skip** rather than fail when fixtures are absent, so `cargo test` on a fresh clone
stays green. `CAST_REQUIRE_PARITY=1` — which CI sets — turns a missing fixture set into a
hard failure, so a silent skip can never be mistaken for a pass.

## Verification status — read this before trusting the crate

Stated precisely, because the distinction matters:

| | status |
|---|---|
| `cargo check --all-targets` | ✅ **passes**, zero errors, zero warnings |
| erf-GELU vs `torch.nn.functional.gelu` | ✅ **verified**, max abs error **9.5e-07** |
| forward-pass algorithm vs PyTorch | ✅ **verified**, max \|Δlogit\| **1.366e-05**, argmax **20/20** |
| `cargo test` actually executed | ❌ **not yet** — see below |

The development sandbox has **no C linker** (`cc`, `gcc`, `clang` all absent), so Rust
code can be type-checked but not linked or run there. The forward pass was instead
validated by transcribing the Rust implementation line-for-line into NumPy and running it
against the same PyTorch fixtures the Rust test consumes — identical tensor layouts,
causal mask, tied head, LayerNorm eps, and GELU formulation. That is strong evidence, not
proof of the compiled artifact.

**First execution happens in CI**, where a linker exists. Until that job is green, treat
"the Rust runs correctly" as expected-but-unproven.

### The GELU trap

PyTorch's default `F.gelu` is the **exact erf** formulation, not the tanh approximation.
Measured on `linspace(-8, 8, 4001)`:

```
erf-based  (used here): max abs err 9.537e-07
tanh approximation:     max abs err 4.735e-04   ← 497x worse, at the tolerance limit
```

Using tanh would sit right at the parity tolerance and compound across 4 layers. If you
touch `gelu()` in `src/model.rs`, re-run the comparison.

## Layout

```
src/format.rs     model.cast reader + FNV-1a + a ~200-line hand-rolled JSON header parser
src/tokenizer.rs  encode/decode, including the three detokenizer rules from docs/LORE.md §III
src/model.rs      the forward pass: pre-norm blocks, causal attention, erf-GELU, tied head
src/lib.rs        the public `Cast` API
tests/parity.rs   the parity harness
tests/json.rs     minimal JSON reader so tests need no dev-dependencies
```

## Conventions that must not drift

- Linear weights keep PyTorch's `(out_features, in_features)` layout → `y = x @ Wᵀ + b`
- LayerNorm `eps = 1e-5`, **biased** variance (divide by N, not N−1)
- Attention softmax with max-subtraction; scale `1/sqrt(head_dim)`
- Dropout is a no-op at inference and is ignored entirely
- Greedy decoding only — one right answer means sampling can only hurt

## Roadmap: npm and PyPI

The core is deliberately binding-agnostic. What each needs:

- **npm (napi-rs)** — a thin `#[napi]` wrapper exposing `load`/`cast`, plus the prebuilt
  matrix (linux-x64-gnu, darwin-x64, darwin-arm64, win32-x64-msvc). The `model.cast` file
  ships as package data or is fetched on first use.
- **PyPI (PyO3/maturin)** — a `#[pymodule]` exposing the same two calls, giving the Python
  package a native fast path that doesn't require torch for *inference*.
- **crates.io** — ready now; it's this crate.

None of these change the core. That's the point of doing the port before the bindings.
