#!/usr/bin/env python3
"""
export_weights.py — freeze a trained nedb-cast-slm checkpoint into a single,
self-contained `model.cast` file that the pure-Rust inference crate can load
with ZERO Python/PyTorch runtime dependency.

Reads
-----
  * a PyTorch checkpoint (default: runs/v1/ckpt.pt) written by cast.train, whose
    top-level dict is {"model": state_dict, "opt": ..., "step": int, "cfg": dict}
  * a tokenizer json (default: data/tokenizer.json) of the form {"itos": [...]}

Writes
------
  * one binary file (default: model.cast) — see FORMAT below.

FORMAT  (little-endian throughout)
----------------------------------
  offset 0    : 8  bytes  magic            = b"CASTMDL1"
  offset 8    : 4  bytes  u32  header_len  = len(header_json_utf8)
  offset 12   : H  bytes  header_json (UTF-8)          -- H == header_len
  offset 12+H : ... f32 tensor blob (concatenated, row-major, C-contiguous)

  The header JSON is:
    {
      "format": "cast-model/1",
      "config": { vocab_size, n_embd, n_layer, n_head, block_size, bias, ... },
      "vocab":  [ "<pad>", "<s>", ... ],          # full itos, len == vocab_size
      "tensors": [
         { "name": "tok_emb.weight",
           "shape": [516, 256],
           "offset": 0,          # byte offset INTO the blob (not the file)
           "length": 528384 },  # byte length == prod(shape)*4
         ...
      ],
      "blob_bytes": <int>,       # total byte length of the blob
      "checksum": { "algo": "fnv1a64", "value": <uint64 as decimal string> }
    }

  Weight tying: `head.weight` IS the same tensor object as `tok_emb.weight` in
  the PyTorch model (verified: shared storage). We therefore export it EXACTLY
  ONCE, under the name "tok_emb.weight", and the Rust loader aliases the head to
  it. No "head.weight" entry appears in the blob.

  Every Linear weight keeps PyTorch's (out_features, in_features) layout, i.e.
  a forward pass is `y = x @ W.T + b`. The Rust code mirrors that convention.

Checksum
--------
  FNV-1a 64-bit over the raw blob bytes. Chosen because it is a handful of lines
  of identical code in both Python (stdlib only) and Rust (no crates), so the
  format stays dependency-free on both sides. It is an integrity check against
  truncation/corruption, not a cryptographic guarantee.

Usage
-----
  python scripts/export_weights.py \
      --ckpt runs/v1/ckpt.pt --tokenizer data/tokenizer.json --out model.cast
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time

MAGIC = b"CASTMDL1"

# The 51 real tensors of a 4-layer model, in a fixed, documented order.
# head.weight is intentionally absent (tied to tok_emb.weight).
_CONFIG_KEYS = ["vocab_size", "n_embd", "n_layer", "n_head", "block_size", "bias"]


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit hash. Identical algorithm implemented in the Rust loader."""
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    mask = 0xFFFFFFFFFFFFFFFF
    for b in data:
        h ^= b
        h = (h * prime) & mask
    return h


def _expected_tensor_names(n_layer: int) -> list:
    names = ["tok_emb.weight", "pos_emb.weight"]
    for i in range(n_layer):
        p = f"blocks.{i}."
        names += [
            p + "ln1.weight", p + "ln1.bias",
            p + "attn.attn.weight", p + "attn.attn.bias",
            p + "attn.proj.weight", p + "attn.proj.bias",
            p + "ln2.weight", p + "ln2.bias",
            p + "mlp.fc.weight", p + "mlp.fc.bias",
            p + "mlp.proj.weight", p + "mlp.proj.bias",
        ]
    names += ["ln_f.weight", "ln_f.bias"]
    return names


def load_checkpoint(path: str, retries: int = 1):
    """torch.load with a single retry, since a training run may be mid-write."""
    import torch  # imported lazily so --help works without torch
    last = None
    for attempt in range(retries + 1):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001 - mid-write can raise many things
            last = e
            if attempt < retries:
                sys.stderr.write(
                    f"[export] torch.load failed ({e!r}); retrying in 5s...\n")
                time.sleep(5)
    raise RuntimeError(f"could not load checkpoint {path!r}: {last!r}")


def export_from_loaded(ck: dict, itos: list, out_path: str,
                       ckpt_path: str = "<in-memory>") -> dict:
    """Export from an ALREADY-LOADED checkpoint dict + itos list.

    Split out from `export` so that dump_parity_fixtures.py can load the (moving)
    checkpoint exactly once and guarantee that model.cast and the fixtures it
    dumps describe bit-identical weights.
    """
    import torch  # lazy

    if "model" not in ck or "cfg" not in ck:
        raise ValueError(
            f"{ckpt_path!r} is not a cast checkpoint (missing 'model'/'cfg')")
    cfg = dict(ck["cfg"])
    sd = ck["model"]
    step = int(ck.get("step", -1))

    vocab_size = int(cfg["vocab_size"])
    if len(itos) != vocab_size:
        raise ValueError(
            f"vocab mismatch: checkpoint {ckpt_path!r} cfg.vocab_size="
            f"{vocab_size} but tokenizer has {len(itos)} tokens. Refusing to "
            f"export a model whose embedding table and vocab disagree.")

    # --- weight tying sanity check -------------------------------------------
    if "head.weight" in sd and "tok_emb.weight" in sd:
        if not torch.equal(sd["head.weight"], sd["tok_emb.weight"]):
            raise ValueError(
                "head.weight != tok_emb.weight — weight tying is NOT holding in "
                "this checkpoint. Aborting rather than exporting an ambiguous "
                "model. (Expected them to be the same tensor.)")

    n_layer = int(cfg["n_layer"])
    expected = _expected_tensor_names(n_layer)
    missing = [n for n in expected if n not in sd]
    if missing:
        raise ValueError(f"checkpoint missing expected tensors: {missing}")

    # --- serialise tensors in fixed order into one blob ----------------------
    blob = bytearray()
    tensors_meta = []
    for name in expected:
        t = sd[name].detach().to(torch.float32).contiguous().cpu()
        arr = t.numpy()
        raw = arr.tobytes(order="C")  # row-major, little-endian f32 on this box
        tensors_meta.append({
            "name": name,
            "shape": list(t.shape),
            "offset": len(blob),
            "length": len(raw),
        })
        blob.extend(raw)

    blob = bytes(blob)
    checksum = fnv1a64(blob)

    header = {
        "format": "cast-model/1",
        "producer": "scripts/export_weights.py",
        "source_ckpt": ckpt_path,
        "source_step": step,
        "config": {k: cfg[k] for k in _CONFIG_KEYS if k in cfg},
        "vocab": itos,
        "tensors": tensors_meta,
        "blob_bytes": len(blob),
        "checksum": {"algo": "fnv1a64", "value": str(checksum)},
    }
    # keep every extra cfg key too (e.g. dropout), but config[] above is the
    # canonical, ordered subset the Rust loader reads.
    for k, v in cfg.items():
        header["config"].setdefault(k, v)

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    with open(out_path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(header_json)))
        fh.write(header_json)
        fh.write(blob)

    return {
        "out": out_path,
        "vocab_size": vocab_size,
        "n_tensors": len(tensors_meta),
        "blob_bytes": len(blob),
        "header_bytes": len(header_json),
        "checksum_fnv1a64": checksum,
        "source_step": step,
        "config": header["config"],
    }


def export(ckpt_path: str, tok_path: str, out_path: str) -> dict:
    """Load a checkpoint from disk and a tokenizer json, then export model.cast."""
    ck = load_checkpoint(ckpt_path)
    with open(tok_path) as fh:
        itos = json.load(fh)["itos"]
    return export_from_loaded(ck, itos, out_path, ckpt_path=ckpt_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a cast checkpoint to model.cast")
    ap.add_argument("--ckpt", default="runs/v1/ckpt.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out", default="model.cast")
    args = ap.parse_args()

    info = export(args.ckpt, args.tokenizer, args.out)
    print(json.dumps(info, indent=2))
    print(f"[export] wrote {info['out']} "
          f"({info['blob_bytes'] + info['header_bytes'] + 12} bytes total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
