//! # nedb-cast-core
//!
//! Pure-Rust inference for **nedb-cast-slm** — a 3.3M-parameter model that casts
//! short English prompts into [NEDB](https://github.com/Eth-Interchained/nedb)
//! query plans (NQL).
//!
//! No Python. No PyTorch. **No dependencies at all.**
//!
//! ```no_run
//! use nedb_cast_core::Cast;
//!
//! let cast = Cast::load("model.cast")?;
//! let nql = cast.cast("top 5 stylists in winter park");
//! assert_eq!(nql, r#"FROM stylists WHERE city = "winter park" LIMIT 5"#);
//! # Ok::<(), nedb_cast_core::LoadError>(())
//! ```
//!
//! Training lives in the Python package (`pip install nedb-cast-slm`); this crate
//! is the inference half, frozen into a single portable `model.cast` file by
//! `scripts/export_weights.py`. That split is deliberate: training needs PyTorch,
//! inference is a few thousand matmuls over a tiny model and needs nothing.
//!
//! ## Numerical parity
//!
//! This crate is held to exact parity with the PyTorch implementation by
//! `tests/parity.rs`, which asserts per-logit agreement within a tight tolerance
//! **and** byte-identical decoded strings across a fixture set. See
//! `rust/README.md` for how to regenerate fixtures and run the check.

mod format;
mod model;
mod tokenizer;

pub use format::{fnv1a64, Config, LoadError, ModelFile};
pub use model::CastModel;
pub use tokenizer::Tokenizer;

use std::path::Path;

/// The high-level API: load once, cast many.
pub struct Cast {
    pub model: CastModel,
    pub tok: Tokenizer,
    /// True when the container carried a checksum and it verified.
    pub checksum_verified: bool,
}

impl Cast {
    /// Load a `model.cast` container produced by `scripts/export_weights.py`.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, LoadError> {
        let mf = ModelFile::load(path)?;
        Self::from_model_file(&mf)
    }

    pub fn from_bytes(raw: &[u8]) -> Result<Self, LoadError> {
        let mf = ModelFile::from_bytes(raw)?;
        Self::from_model_file(&mf)
    }

    fn from_model_file(mf: &ModelFile) -> Result<Self, LoadError> {
        let model = CastModel::from_file(mf)?;
        let tok = Tokenizer::new(mf.vocab.clone());
        Ok(Cast { model, tok, checksum_verified: mf.checksum_ok })
    }

    /// Cast a prompt into NQL text. Greedy decoding — for a DSL there is exactly
    /// one right answer, so sampling can only hurt.
    pub fn cast(&self, prompt: &str) -> String {
        self.cast_with_limit(prompt, 72)
    }

    pub fn cast_with_limit(&self, prompt: &str, max_new_tokens: usize) -> String {
        let ids = self.tok.encode_prompt(prompt);
        let gen = self.model.generate(&ids, max_new_tokens, self.tok.eos_id);
        self.tok.decode(&gen)
    }

    /// Raw logits at the final position — useful for parity checks and for
    /// callers that want to inspect confidence.
    pub fn logits(&self, prompt: &str) -> Vec<f32> {
        let ids = self.tok.encode_prompt(prompt);
        self.model.forward(&ids, true)
    }

    /// Greedy token ids without detokenising.
    pub fn cast_ids(&self, prompt: &str, max_new_tokens: usize) -> Vec<u32> {
        let ids = self.tok.encode_prompt(prompt);
        self.model.generate(&ids, max_new_tokens, self.tok.eos_id)
    }

    pub fn vocab_size(&self) -> usize {
        self.tok.len()
    }

    pub fn n_params(&self) -> usize {
        let c = &self.model.cfg;
        let per_block = 3 * c.n_embd * c.n_embd // qkv
            + c.n_embd * c.n_embd               // attn proj
            + 4 * c.n_embd * c.n_embd           // mlp fc
            + 4 * c.n_embd * c.n_embd;          // mlp proj
        c.vocab_size * c.n_embd + c.block_size * c.n_embd + c.n_layer * per_block
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_matches_known_vector() {
        // FNV-1a 64 of b"" and b"a" are fixed, well-known values.
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn detok_merges_dates_and_floats() {
        // regression: digit-run merge must include '-' and '.'
        let toks = vec![
            "from", "products", "valid", "as", "of", "\"", "2", "0", "2", "6", "-", "1", "0",
            "-", "1", "8", "\"", "limit", "3", "1", ".", "0",
        ];
        let s = Tokenizer::detok(&toks);
        assert!(s.contains("\"2026-10-18\""), "got: {s}");
        assert!(s.contains("31.0"), "got: {s}");
    }

    #[test]
    fn detok_does_not_uppercase_keywords_inside_quotes() {
        // regression: SEARCH "rate limit" must not become "rate LIMIT"
        let toks = vec!["from", "events", "search", "\"", "rate", "limit", "\"", "limit", "5"];
        let s = Tokenizer::detok(&toks);
        assert!(s.contains("SEARCH \"rate limit\""), "got: {s}");
        assert!(s.trim_end().ends_with("LIMIT 5"), "got: {s}");
    }

    #[test]
    fn pre_tokenize_splits_digits_and_ops() {
        let t = Tokenizer::pre_tokenize("orders total >= 99.5");
        assert_eq!(t, vec!["orders", "total", ">=", "9", "9", ".", "5"]);
    }
}
