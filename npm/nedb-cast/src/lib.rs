//! Node.js bindings for `nedb-cast-slm`.
//!
//! The core is pure Rust with zero dependencies, so this wrapper is thin by
//! design: it forwards to the same code the Rust crate and the CI parity test
//! exercise, which means Node gets byte-identical output to Python and Rust
//! rather than a third implementation that happens to agree today.
//!
//! ```js
//! const { Cast } = require('@interchained/cast');
//! const cast = Cast.load('model.cast');
//! cast.cast('top 5 stylists in winter park');
//! // FROM stylists WHERE city = "winter park" LIMIT 5
//! ```

#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

use nedb_cast_slm::Cast as CoreCast;

#[napi]
pub struct Cast {
    inner: CoreCast,
}

#[napi]
impl Cast {
    /// Load a `model.cast` container from disk.
    ///
    /// Download it from the GitHub release; it is the same file the Python
    /// package and the Rust crate consume.
    #[napi(factory)]
    pub fn load(path: String) -> Result<Self> {
        let inner = CoreCast::load(&path)
            .map_err(|e| Error::new(Status::GenericFailure, format!("{e}")))?;
        Ok(Cast { inner })
    }

    /// Load from an in-memory buffer, for callers that fetch the weights
    /// themselves (bundlers, serverless, embedded assets).
    #[napi(factory)]
    pub fn from_buffer(buf: Buffer) -> Result<Self> {
        let inner = CoreCast::from_bytes(&buf)
            .map_err(|e| Error::new(Status::GenericFailure, format!("{e}")))?;
        Ok(Cast { inner })
    }

    /// Cast a short prompt into NQL text. Greedy decoding — for a DSL there is
    /// exactly one right answer, so sampling can only hurt.
    #[napi]
    pub fn cast(&self, prompt: String) -> String {
        self.inner.cast(&prompt)
    }

    #[napi]
    pub fn cast_with_limit(&self, prompt: String, max_new_tokens: u32) -> String {
        self.inner.cast_with_limit(&prompt, max_new_tokens as usize)
    }

    /// Raw logits at the final position, for callers that want confidence.
    #[napi]
    pub fn logits(&self, prompt: String) -> Vec<f64> {
        self.inner.logits(&prompt).into_iter().map(|v| v as f64).collect()
    }

    #[napi(getter)]
    pub fn vocab_size(&self) -> u32 {
        self.inner.vocab_size() as u32
    }

    #[napi(getter)]
    pub fn n_params(&self) -> u32 {
        self.inner.n_params() as u32
    }

    /// True when the container carried a checksum and it verified. A corrupt
    /// model would emit plausible-but-wrong queries, so callers should check.
    #[napi(getter)]
    pub fn checksum_verified(&self) -> bool {
        self.inner.checksum_verified
    }
}
