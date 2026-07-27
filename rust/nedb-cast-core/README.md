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

## It ships inside NEDB

As of **nedb-engine v2.8.0** this crate is a feature of the database daemon itself:

```bash
cargo install nedb-engine --features cast
nedbd --dag --cast ./data

curl -X POST localhost:7070/v1/databases/shop/cast -d '{"prompt":"orders over 100"}'
# {"nql":"FROM orders WHERE total > 100","valid":true,"collection_known":true}
```

The engine wraps `Cast` in a `Caster` that also checks the generated plan against the
collections the database actually has — a plan naming a collection that does not exist
returns 422 with the reason, never an empty result set. That check is the reason the
planner belongs in the engine: schema knowledge is free there and stale everywhere else.

`nedb-engine` depends on this crate optionally (`cast = ["dep:nedb-cast-slm"]`), so a
default build pulls none of it. And since this crate has an empty dependency tree, the
feature adds exactly one crate to the graph.

```rust
use nedb_engine::cast::Caster;

let caster = Caster::load(&data_dir)?;
let out = caster.cast_checked("orders over 100", &db.id_index.collections());
if out.collection_known && nedb_engine::nql::parse(&out.nql).is_ok() {
    let (rows, count) = nedb_engine::nql::query(&db, &out.nql)?;
}
```

## Weights

Weights ship as GitHub **release assets**, not in the crate. Download `model.cast` from
[the releases page](https://github.com/aiassistsecure/nedb-cast-slm/releases) and pass its
path to `Cast::load`. It is the same container the Python package consumes, so both
languages load byte-identical weights.

## Docs

Full format spec, architecture notes, and the development lore (five bugs, not one of which
was in the model) live in the [repository](https://github.com/aiassistsecure/nedb-cast-slm):
`rust/README.md`, `docs/ARCHITECTURE.md`, `docs/LORE.md`.

## License

BUSL-1.1 · [Interchained LLC](https://interchained.org)
