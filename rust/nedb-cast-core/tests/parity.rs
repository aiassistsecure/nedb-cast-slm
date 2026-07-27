//! Numerical parity against the PyTorch reference implementation.
//!
//! This is the load-bearing test of the whole port. A Rust forward pass that is
//! merely *close* to PyTorch is wrong: greedy decoding picks an argmax, so a
//! 1e-3 drift flips any near-tie and silently produces a different query.
//!
//! Fixtures are produced by `scripts/dump_parity_fixtures.py`, which exports
//! `model.cast` and the expected outputs from a SINGLE `torch.load` so the
//! weights behind the logits are guaranteed identical to the weights under test.
//!
//! Run:
//! ```sh
//! python scripts/dump_parity_fixtures.py \
//!     --ckpt runs/v1/ckpt.pt --tokenizer data/tokenizer.json \
//!     --model-out rust/nedb-cast-core/tests/model.cast \
//!     --fixtures-out rust/nedb-cast-core/tests/fixtures.json
//! cd rust/nedb-cast-core && cargo test --release
//! ```
//!
//! The tests SKIP (rather than fail) when the fixtures or model are absent, so a
//! plain `cargo test` on a fresh clone stays green. CI generates them first, so
//! CI enforces parity for real. `strict_parity_fixtures_present` fails loudly if
//! `CAST_REQUIRE_PARITY=1`, which CI sets — that way a silent skip can never be
//! mistaken for a pass.

use std::path::{Path, PathBuf};

use nedb_cast_core::Cast;

mod json;
use json::JsonValue;

fn tests_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("CAST_MODEL") {
        return PathBuf::from(p);
    }
    tests_dir().join("model.cast")
}

fn fixtures_path() -> PathBuf {
    if let Ok(p) = std::env::var("CAST_FIXTURES") {
        return PathBuf::from(p);
    }
    tests_dir().join("fixtures.json")
}

fn require_parity() -> bool {
    std::env::var("CAST_REQUIRE_PARITY").map(|v| v == "1").unwrap_or(false)
}

fn load_fixtures() -> Option<JsonValue> {
    let p = fixtures_path();
    let raw = std::fs::read_to_string(&p).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    JsonValue::parse(&raw).ok()
}

/// Guard: in CI (CAST_REQUIRE_PARITY=1) a missing fixture set is a FAILURE, not
/// a skip. Without this, a broken fixture-generation step would look like a pass.
#[test]
fn strict_parity_fixtures_present() {
    if !require_parity() {
        eprintln!("CAST_REQUIRE_PARITY unset — skipping strictness guard");
        return;
    }
    assert!(
        model_path().exists(),
        "CAST_REQUIRE_PARITY=1 but {} is missing",
        model_path().display()
    );
    let fx = load_fixtures().expect("fixtures.json missing or unparseable under CAST_REQUIRE_PARITY=1");
    let n = fx.get("fixtures").and_then(|f| f.as_array()).map(|a| a.len()).unwrap_or(0);
    assert!(n >= 10, "expected >=10 parity fixtures, found {n}");
}

#[test]
fn container_loads_and_checksum_verifies() {
    let mp = model_path();
    if !mp.exists() {
        eprintln!("skip: {} not present", mp.display());
        return;
    }
    let cast = Cast::load(&mp).expect("failed to load model.cast");
    assert!(cast.checksum_verified, "container checksum did not verify");
    assert!(cast.vocab_size() > 100, "implausible vocab: {}", cast.vocab_size());
}

/// The strong numerical check: every logit at the final prompt position.
#[test]
fn logits_match_pytorch() {
    let mp = model_path();
    let Some(fx) = load_fixtures() else {
        eprintln!("skip: no fixtures");
        return;
    };
    if !mp.exists() {
        eprintln!("skip: no model.cast");
        return;
    }
    let cast = Cast::load(&mp).expect("load model.cast");
    let tol: f32 = std::env::var("CAST_PARITY_TOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-4);

    let items = fx.get("fixtures").and_then(|f| f.as_array()).expect("fixtures array");
    let mut worst = 0f32;
    let mut worst_prompt = String::new();
    let mut checked = 0usize;

    for it in items {
        let prompt = it.get("prompt").and_then(|v| v.as_str()).expect("prompt");
        let want = it
            .get("prompt_logits")
            .and_then(|v| v.as_f32_array())
            .expect("prompt_logits");

        // Verify our tokenization agrees before comparing numbers — a token
        // mismatch would otherwise masquerade as numerical drift.
        if let Some(want_ids) = it.get("input_ids").and_then(|v| v.as_u32_array()) {
            let got_ids = cast.tok.encode_prompt(prompt);
            assert_eq!(
                got_ids, want_ids,
                "tokenization mismatch for {prompt:?}\n  rust: {got_ids:?}\n  py:   {want_ids:?}"
            );
        }

        let got = cast.logits(prompt);
        assert_eq!(
            got.len(),
            want.len(),
            "logit length mismatch for {prompt:?}: rust {} vs py {}",
            got.len(),
            want.len()
        );
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (g - w).abs();
            if d > worst {
                worst = d;
                worst_prompt = format!("{prompt:?} @ logit {i}");
            }
        }
        checked += 1;
    }

    eprintln!("parity: {checked} prompts, max |Δlogit| = {worst:.3e} (tol {tol:.1e})");
    assert!(
        worst <= tol,
        "logit drift {worst:.3e} exceeds tolerance {tol:.1e} at {worst_prompt}"
    );
}

/// The user-facing check: identical token ids AND identical decoded strings.
#[test]
fn decoded_output_matches_pytorch_exactly() {
    let mp = model_path();
    let Some(fx) = load_fixtures() else {
        eprintln!("skip: no fixtures");
        return;
    };
    if !mp.exists() {
        eprintln!("skip: no model.cast");
        return;
    }
    let cast = Cast::load(&mp).expect("load model.cast");
    let items = fx.get("fixtures").and_then(|f| f.as_array()).expect("fixtures array");

    let mut mismatches: Vec<String> = Vec::new();
    for it in items {
        let prompt = it.get("prompt").and_then(|v| v.as_str()).expect("prompt");
        let want_nql = it.get("nql").and_then(|v| v.as_str()).unwrap_or("");
        let max_new = fx.get("max_new_tokens").and_then(|v| v.as_usize()).unwrap_or(72);

        if let Some(want_ids) = it.get("gen_ids").and_then(|v| v.as_u32_array()) {
            let got_ids = cast.cast_ids(prompt, max_new);
            if got_ids != want_ids {
                mismatches.push(format!(
                    "gen_ids differ for {prompt:?}\n    rust: {got_ids:?}\n    py:   {want_ids:?}"
                ));
                continue;
            }
        }
        let got = cast.cast_with_limit(prompt, max_new);
        if got != want_nql {
            mismatches.push(format!(
                "nql differs for {prompt:?}\n    rust: {got:?}\n    py:   {want_nql:?}"
            ));
        }
    }

    if !mismatches.is_empty() {
        panic!(
            "{} of {} fixtures disagreed:\n{}",
            mismatches.len(),
            items.len(),
            mismatches.join("\n")
        );
    }
    eprintln!("decode parity: {} / {} exact", items.len(), items.len());
}
