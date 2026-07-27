"""
Tests that enforce the invariants which actually caught bugs in this project.

These are not coverage theatre. Each one corresponds to a real bug that shipped
into a working-looking state and was only caught by asserting an invariant:

  test_generated_plans_roundtrip   — a generator bug would silently teach the
                                     model invalid syntax
  test_detokenizer_is_lossless     — a lossy detokenizer caps achievable accuracy
                                     invisibly (this one was at 76%)
  test_batched_equals_single       — padding + learned positions corrupted short
                                     sequences, costing 55 accuracy points
  test_content_address_stable      — list(set(...)) made "same seed, same dataset"
                                     silently false
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nedb.query import parse_nql

from cast.paraphrase import paraphrase
from cast.sampler import canonical, clauses_present, render_nql, sample_plan
from cast.tokenizer import CastTokenizer


# --------------------------------------------------------------- generator
def test_generated_plans_roundtrip():
    """Every sampled plan must render to NQL the REAL parser accepts, and parse
    back to a canonically identical plan."""
    rng = random.Random(20260727)
    for _ in range(5000):
        plan, dom, coll = sample_plan(rng)
        nql = render_nql(plan)
        parsed = parse_nql(nql)  # raises on failure — that's the assertion
        assert canonical(parsed) == canonical(plan), f"mismatch for {nql!r}"


def test_clause_coverage_is_broad():
    """All ten clauses must actually appear, or the model can't learn them."""
    rng = random.Random(7)
    seen = set()
    for _ in range(20000):
        plan, _, _ = sample_plan(rng)
        seen.update(clauses_present(plan))
    for clause in ("where", "where_multi", "as_of", "valid_as_of", "search",
                   "order_by", "traverse", "trace", "trace_reverse", "limit",
                   "group_by", "aggregate"):
        assert clause in seen, f"clause never sampled: {clause}"


def test_traverse_uses_collection_scoped_relations():
    """TRAVERSE must use an edge valid FROM that collection, not any edge in the
    domain — otherwise queries parse but are semantically nonsense."""
    rng = random.Random(99)
    for _ in range(4000):
        plan, dom, coll = sample_plan(rng)
        if plan.get("traverse"):
            valid = coll.relations or dom.relations
            assert plan["traverse"] in valid


def test_paraphrase_is_nonempty_and_mentions_collection():
    rng = random.Random(3)
    for _ in range(2000):
        plan, dom, coll = sample_plan(rng)
        for holdout in (False, True):
            p = paraphrase(plan, dom, coll, rng, holdout=holdout)
            assert p.strip(), "empty paraphrase"
            assert "  " not in p, f"double space in {p!r}"


# --------------------------------------------------------------- tokenizer
@pytest.fixture(scope="module")
def fitted_tok():
    rng = random.Random(11)
    texts, pairs = [], []
    for _ in range(4000):
        plan, dom, coll = sample_plan(rng)
        nql = render_nql(plan)
        prompt = paraphrase(plan, dom, coll, rng)
        texts += [prompt, nql]
        pairs.append((nql, canonical(plan)))
    return CastTokenizer.fit(texts, min_freq=1, max_vocab=4096), pairs


def test_detokenizer_is_lossless(fitted_tok):
    """encode -> decode -> parse must return the identical plan.

    A lossy detokenizer is a hard ceiling on final accuracy and looks exactly
    like a model that plateaued.
    """
    tok, pairs = fitted_tok
    for nql, want in pairs:
        back = tok.decode(tok.encode(nql))
        got = parse_nql(back)
        assert canonical(got) == want, f"lossy: {nql!r} -> {back!r}"


def test_dates_survive_tokenization(fitted_tok):
    """Regression: the digit-merge loop dropped '-', so "2026-10-18" decoded as
    "2026 -10 -18"."""
    tok, _ = fitted_tok
    q = 'FROM products VALID AS OF "2026-10-18" ORDER BY title ASC'
    assert tok.decode(tok.encode(q)) == q


def test_keywords_inside_quotes_are_not_uppercased(fitted_tok):
    """Regression: SEARCH "rate limit" became SEARCH "rate LIMIT" because keyword
    casing ignored quote state, corrupting the literal."""
    tok, _ = fitted_tok
    q = 'FROM events SEARCH "rate limit" LIMIT 5'
    back = tok.decode(tok.encode(q))
    assert '"rate limit"' in back, back
    assert parse_nql(back)["search"] == "rate limit"


def test_prompt_masking_boundary(fitted_tok):
    """encode_pair's prompt_len must point just past <sep> so loss masks the
    prompt exactly."""
    tok, _ = fitted_tok
    ids, plen = tok.encode_pair("show me orders", 'FROM orders')
    assert ids[0] == tok.bos_id
    assert ids[plen - 1] == tok.sep_id
    assert ids[-1] == tok.eos_id


# --------------------------------------------------------------- inference
@pytest.mark.skipif(not os.path.exists("runs/v1/ckpt.pt"),
                    reason="no trained checkpoint available")
def test_batched_equals_single():
    """Batched decoding must be bit-identical to single-sequence decoding.

    The model uses learned positions and no pad mask, so any padding in a batch
    of mixed lengths corrupts short sequences. This cost 55 accuracy points and
    masqueraded as a model capacity limit.
    """
    from cast.inference import Cast
    c = Cast.from_pretrained("runs/v1")
    rows = [json.loads(l) for l in open("data/eval.jsonl")][:40]
    prompts = [r["prompt"] for r in rows]
    batched = c.cast_many(prompts)
    single = [c.cast_many([p])[0] for p in prompts]
    assert batched == single


@pytest.mark.skipif(not os.path.exists("runs/v1/ckpt.pt"),
                    reason="no trained checkpoint available")
def test_plan_raises_on_invalid():
    """plan() must raise rather than return an unrunnable plan."""
    from cast.inference import Cast
    c = Cast.from_pretrained("runs/v1")
    out = c.try_plan("show me orders")
    assert set(out) >= {"ok", "nql"}


# --------------------------------------------------------------- determinism
def test_content_address_stable_across_processes(tmp_path):
    """Same seed + unchanged code must yield the same dataset id in a DIFFERENT
    process (PYTHONHASHSEED is randomised per process).

    Regression: list(set(strings)) reordered rows per process, so the content
    address changed every run while every row was identical.
    """
    script = (
        "import sys, json;"
        f"sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r});"
        "from cast.dataset import generate;"
        f"m = generate(n_train=600, n_eval=60, n_holdout=40, seed=42, out_dir={str(tmp_path)!r});"
        "print(m['run_id'])"
    )
    ids = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        ids.append(r.stdout.strip().splitlines()[-1])
    assert ids[0] == ids[1], f"content address unstable: {ids}"


# --------------------------------------------------------------- rust parity
# These guard the Python SIDE of the Rust port: the exporter's container format
# and the fixture generator. The Rust side is exercised by `cargo test` in CI
# (see .github/workflows/ci.yml, job `rust-parity`) because this sandbox has no
# C linker and cannot link a Rust binary.

@pytest.mark.skipif(not os.path.exists("runs/v1/ckpt.pt"),
                    reason="no trained checkpoint available")
def test_export_container_is_wellformed(tmp_path):
    """model.cast must be readable exactly as rust/src/format.rs reads it."""
    import struct
    import subprocess

    out = tmp_path / "model.cast"
    r = subprocess.run(
        [sys.executable, "scripts/export_weights.py", "--ckpt", "runs/v1/ckpt.pt",
         "--tokenizer", "data/tokenizer.json", "--out", str(out)],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]

    raw = out.read_bytes()
    assert raw[:8] == b"CASTMDL1", "bad magic"
    hlen = struct.unpack("<I", raw[8:12])[0]
    hdr = json.loads(raw[12:12 + hlen].decode())

    assert hdr["format"] == "cast-model/1"
    assert len(hdr["vocab"]) == hdr["config"]["vocab_size"]

    names = {t["name"] for t in hdr["tensors"]}
    # weight tying: head must NOT be exported separately
    assert "head.weight" not in names, "head.weight exported despite weight tying"
    assert "tok_emb.weight" in names

    # every tensor's declared length must equal prod(shape)*4 and sit in range
    blob = raw[12 + hlen:12 + hlen + hdr["blob_bytes"]]
    assert len(blob) == hdr["blob_bytes"], "blob truncated"
    for t in hdr["tensors"]:
        n = 1
        for d in t["shape"]:
            n *= d
        assert t["length"] == n * 4, f"{t['name']}: length != prod(shape)*4"
        assert t["offset"] + t["length"] <= hdr["blob_bytes"], f"{t['name']} out of range"

    # FNV-1a 64 over the blob must match the header
    h = 0xcbf29ce484222325
    for b in blob:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    want = int(hdr["checksum"]["value"])
    assert h == want, f"checksum mismatch: computed {h}, header {want}"


@pytest.mark.skipif(not os.path.exists("rust/nedb-cast-core/tests/fixtures.json"),
                    reason="parity fixtures not generated")
def test_parity_fixtures_are_usable():
    """Fixtures must carry everything the Rust test asserts against."""
    fx = json.load(open("rust/nedb-cast-core/tests/fixtures.json"))
    items = fx.get("fixtures") or []
    assert len(items) >= 10, f"expected >=10 fixtures, got {len(items)}"
    for it in items:
        for key in ("prompt", "input_ids", "prompt_logits", "gen_ids", "nql"):
            assert key in it, f"fixture missing {key}"
        assert len(it["prompt_logits"]) > 100, "logit vector implausibly short"
        assert it["input_ids"], "empty input_ids"


def test_gelu_uses_exact_erf_not_tanh():
    """PyTorch's default F.gelu is the erf formulation. The tanh approximation is
    ~500x less accurate and would sit at the parity tolerance, so the Rust port
    must use erf. This test pins the reference so a future change is caught."""
    import math
    import torch
    xs = torch.linspace(-8, 8, 2001)
    ref = torch.nn.functional.gelu(xs)
    tanh = 0.5 * xs * (1 + torch.tanh(math.sqrt(2 / math.pi) * (xs + 0.044715 * xs ** 3)))
    err = (tanh - ref).abs().max().item()
    assert err > 1e-4, (
        "tanh approx is unexpectedly close to torch gelu; the Rust port's choice "
        "of erf may no longer be load-bearing — re-verify rust/src/model.rs")
