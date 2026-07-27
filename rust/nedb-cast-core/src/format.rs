//! Reader for the `model.cast` container written by `scripts/export_weights.py`.
//!
//! Layout (little-endian throughout):
//!
//! ```text
//!   0        : 8 bytes  magic = b"CASTMDL1"
//!   8        : 4 bytes  u32 header_len
//!   12       : H bytes  header JSON (UTF-8)
//!   12 + H   : ...      f32 blob, concatenated, row-major
//! ```
//!
//! The header carries the config, the full vocabulary, and a tensor directory of
//! `{name, shape, offset, length}` where `offset` is relative to the START OF THE
//! BLOB, not the file.
//!
//! We hand-roll a tiny JSON reader rather than take a `serde` dependency. The
//! header shape is fixed and known, so a permissive scanner is sufficient and it
//! keeps the crate's dependency tree genuinely empty.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"CASTMDL1";

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub block_size: usize,
    pub bias: bool,
}

/// A borrowed view of one tensor inside the blob.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub offset: usize,
    pub length: usize,
}

pub struct ModelFile {
    pub config: Config,
    pub vocab: Vec<String>,
    pub tensors: HashMap<String, TensorInfo>,
    pub blob: Vec<f32>,
    pub checksum_ok: bool,
}

#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    BadMagic,
    Truncated(&'static str),
    Header(String),
    MissingTensor(String),
    ShapeMismatch { name: String, want: Vec<usize>, got: Vec<usize> },
    Checksum { want: u64, got: u64 },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "io error: {e}"),
            LoadError::BadMagic => write!(f, "not a model.cast file (bad magic)"),
            LoadError::Truncated(w) => write!(f, "file truncated while reading {w}"),
            LoadError::Header(m) => write!(f, "malformed header: {m}"),
            LoadError::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            LoadError::ShapeMismatch { name, want, got } => {
                write!(f, "tensor {name} shape mismatch: want {want:?}, got {got:?}")
            }
            LoadError::Checksum { want, got } => {
                write!(f, "blob checksum mismatch: want {want}, got {got}")
            }
        }
    }
}

impl std::error::Error for LoadError {}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        LoadError::Io(e)
    }
}

/// FNV-1a 64-bit. Matches the Python exporter byte for byte.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

impl ModelFile {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, LoadError> {
        let raw = fs::read(path)?;
        Self::from_bytes(&raw)
    }

    pub fn from_bytes(raw: &[u8]) -> Result<Self, LoadError> {
        if raw.len() < 12 {
            return Err(LoadError::Truncated("magic/header length"));
        }
        if &raw[0..8] != MAGIC {
            return Err(LoadError::BadMagic);
        }
        let hlen = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
        let hstart = 12;
        let hend = hstart + hlen;
        if raw.len() < hend {
            return Err(LoadError::Truncated("header json"));
        }
        let header = std::str::from_utf8(&raw[hstart..hend])
            .map_err(|e| LoadError::Header(format!("header not utf-8: {e}")))?;

        let config = parse_config(header)?;
        let vocab = parse_vocab(header)?;
        let tensors = parse_tensors(header)?;
        let blob_bytes = json_usize(header, "blob_bytes")
            .ok_or_else(|| LoadError::Header("blob_bytes missing".into()))?;

        let bstart = hend;
        let bend = bstart + blob_bytes;
        if raw.len() < bend {
            return Err(LoadError::Truncated("tensor blob"));
        }
        let blob_raw = &raw[bstart..bend];

        // integrity: FNV-1a over the raw blob. A container without a checksum
        // still loads, but `checksum_ok` reports that it was unverified rather
        // than silently implying it passed.
        let checksum_ok = match json_u64_in_object(header, "checksum", "value") {
            Some(want) => {
                let got = fnv1a64(blob_raw);
                if got != want {
                    return Err(LoadError::Checksum { want, got });
                }
                true
            }
            None => false,
        };

        // decode f32 little-endian
        if blob_bytes % 4 != 0 {
            return Err(LoadError::Header("blob_bytes not a multiple of 4".into()));
        }
        let mut blob = Vec::with_capacity(blob_bytes / 4);
        for c in blob_raw.chunks_exact(4) {
            blob.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }

        if vocab.len() != config.vocab_size {
            return Err(LoadError::Header(format!(
                "vocab len {} != config.vocab_size {}",
                vocab.len(),
                config.vocab_size
            )));
        }

        Ok(ModelFile { config, vocab, tensors, blob, checksum_ok })
    }

    /// Fetch a tensor as an f32 slice, asserting the expected shape.
    pub fn tensor(&self, name: &str, want: &[usize]) -> Result<&[f32], LoadError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| LoadError::MissingTensor(name.to_string()))?;
        if info.shape != want {
            return Err(LoadError::ShapeMismatch {
                name: name.to_string(),
                want: want.to_vec(),
                got: info.shape.clone(),
            });
        }
        let start = info.offset / 4;
        let n = info.length / 4;
        if start + n > self.blob.len() {
            return Err(LoadError::Truncated("tensor slice out of range"));
        }
        Ok(&self.blob[start..start + n])
    }
}

// ---------------------------------------------------------------- tiny JSON

/// Find `"key"` at any depth and return the byte index just past its colon.
fn find_key(s: &str, key: &str) -> Option<usize> {
    let pat = format!("\"{key}\"");
    let mut from = 0usize;
    while let Some(i) = s[from..].find(&pat) {
        let abs = from + i;
        let after = abs + pat.len();
        let rest = &s[after..];
        let trimmed = rest.trim_start();
        if trimmed.starts_with(':') {
            let colon = after + (rest.len() - trimmed.len()) + 1;
            return Some(colon);
        }
        from = after;
    }
    None
}

fn json_usize(s: &str, key: &str) -> Option<usize> {
    let at = find_key(s, key)?;
    let rest = s[at..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn json_bool(s: &str, key: &str) -> Option<bool> {
    let at = find_key(s, key)?;
    let rest = s[at..].trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// Read `{ "obj": { ... "field": <number or "number"> ... } }`.
fn json_u64_in_object(s: &str, obj: &str, field: &str) -> Option<u64> {
    let at = find_key(s, obj)?;
    let rest = &s[at..];
    let open = rest.find('{')?;
    let close = rest[open..].find('}')? + open;
    let inner = &rest[open..=close];
    let fat = find_key(inner, field)?;
    let v = inner[fat..].trim_start();
    // exporter may emit the value as a JSON number or a decimal string
    let v = v.trim_start_matches('"');
    let end = v.find(|c: char| !c.is_ascii_digit()).unwrap_or(v.len());
    v[..end].parse().ok()
}

fn parse_config(s: &str) -> Result<Config, LoadError> {
    let at = find_key(s, "config").ok_or_else(|| LoadError::Header("config missing".into()))?;
    let rest = &s[at..];
    let open = rest.find('{').ok_or_else(|| LoadError::Header("config not an object".into()))?;
    let close = rest[open..]
        .find('}')
        .ok_or_else(|| LoadError::Header("config unterminated".into()))?
        + open;
    let inner = &rest[open..=close];
    Ok(Config {
        vocab_size: json_usize(inner, "vocab_size")
            .ok_or_else(|| LoadError::Header("vocab_size missing".into()))?,
        n_embd: json_usize(inner, "n_embd")
            .ok_or_else(|| LoadError::Header("n_embd missing".into()))?,
        n_layer: json_usize(inner, "n_layer")
            .ok_or_else(|| LoadError::Header("n_layer missing".into()))?,
        n_head: json_usize(inner, "n_head")
            .ok_or_else(|| LoadError::Header("n_head missing".into()))?,
        block_size: json_usize(inner, "block_size")
            .ok_or_else(|| LoadError::Header("block_size missing".into()))?,
        bias: json_bool(inner, "bias").unwrap_or(true),
    })
}

/// Decode a JSON string literal (handles \" \\ \/ \n \r \t \uXXXX).
fn decode_json_string(src: &str) -> (String, usize) {
    let b = src.as_bytes();
    let mut out = String::new();
    let mut i = 0usize; // src[0] is the opening quote
    debug_assert_eq!(b[0], b'"');
    i += 1;
    while i < b.len() {
        match b[i] {
            b'"' => return (out, i + 1),
            b'\\' => {
                i += 1;
                if i >= b.len() {
                    break;
                }
                match b[i] {
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'b' => out.push('\u{8}'),
                    b'f' => out.push('\u{c}'),
                    b'u' => {
                        let hex = &src[i + 1..i + 5];
                        if let Ok(cp) = u32::from_str_radix(hex, 16) {
                            if let Some(ch) = char::from_u32(cp) {
                                out.push(ch);
                            }
                        }
                        i += 4;
                    }
                    other => out.push(other as char),
                }
                i += 1;
            }
            _ => {
                // copy one UTF-8 scalar
                let ch = src[i..].chars().next().unwrap();
                out.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    (out, i)
}

fn parse_vocab(s: &str) -> Result<Vec<String>, LoadError> {
    let at = find_key(s, "vocab").ok_or_else(|| LoadError::Header("vocab missing".into()))?;
    let rest = &s[at..];
    let open = rest.find('[').ok_or_else(|| LoadError::Header("vocab not an array".into()))?;
    let mut i = open + 1;
    let mut out = Vec::new();
    loop {
        let tail = &rest[i..];
        let t = tail.trim_start();
        let skipped = tail.len() - t.len();
        i += skipped;
        if t.starts_with(']') {
            break;
        }
        if t.starts_with(',') {
            i += 1;
            continue;
        }
        if !t.starts_with('"') {
            return Err(LoadError::Header("vocab entry not a string".into()));
        }
        let (tok, used) = decode_json_string(&rest[i..]);
        out.push(tok);
        i += used;
    }
    Ok(out)
}

fn parse_tensors(s: &str) -> Result<HashMap<String, TensorInfo>, LoadError> {
    let at = find_key(s, "tensors").ok_or_else(|| LoadError::Header("tensors missing".into()))?;
    let rest = &s[at..];
    let open = rest.find('[').ok_or_else(|| LoadError::Header("tensors not an array".into()))?;
    let mut out = HashMap::new();
    let mut i = open + 1;
    loop {
        let tail = &rest[i..];
        let Some(ob) = tail.find('{') else { break };
        // an entry must start before the array closes
        if let Some(cb) = tail.find(']') {
            if cb < ob {
                break;
            }
        }
        let start = i + ob;
        let Some(rel_close) = rest[start..].find('}') else { break };
        let end = start + rel_close;
        let entry = &rest[start..=end];

        let nat = find_key(entry, "name")
            .ok_or_else(|| LoadError::Header("tensor entry has no name".into()))?;
        let nrest = entry[nat..].trim_start();
        if !nrest.starts_with('"') {
            return Err(LoadError::Header("tensor name not a string".into()));
        }
        let (name, _) = decode_json_string(nrest);

        let offset = json_usize(entry, "offset")
            .ok_or_else(|| LoadError::Header(format!("{name}: offset missing")))?;
        let length = json_usize(entry, "length")
            .ok_or_else(|| LoadError::Header(format!("{name}: length missing")))?;

        // shape array
        let sat = find_key(entry, "shape")
            .ok_or_else(|| LoadError::Header(format!("{name}: shape missing")))?;
        let srest = &entry[sat..];
        let so = srest.find('[').ok_or_else(|| LoadError::Header("shape not array".into()))?;
        let sc = srest[so..].find(']').ok_or_else(|| LoadError::Header("shape unterminated".into()))?
            + so;
        let mut shape = Vec::new();
        for part in srest[so + 1..sc].split(',') {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            shape.push(
                p.parse::<usize>()
                    .map_err(|_| LoadError::Header(format!("{name}: bad shape entry {p}")))?,
            );
        }

        out.insert(name, TensorInfo { shape, offset, length });
        i = end + 1;
    }
    Ok(out)
}
