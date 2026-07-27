//! Rust port of `cast/tokenizer.py`. This must match the Python side EXACTLY —
//! it is part of the model contract, not an implementation detail.
//!
//! Encoding: whitespace/punctuation split, lowercased, numbers split into
//! individual digits so any literal is representable with 14 symbols.
//!
//! Decoding has three rules that were genuine bugs in the Python version (see
//! `docs/LORE.md` §III) and are reproduced deliberately here:
//!
//!   1. digit runs re-merge INCLUDING `-` and `.` when a digit follows, so
//!      `2026-10-18` and `31.0` survive round-tripping.
//!   2. NQL keyword uppercasing applies OUTSIDE QUOTES ONLY, so
//!      `SEARCH "rate limit"` does not become `SEARCH "rate LIMIT"`.
//!   3. quoted literals get their inner whitespace tightened.

use std::collections::HashMap;

pub const PAD: &str = "<pad>";
pub const BOS: &str = "<s>";
pub const EOS: &str = "</s>";
pub const UNK: &str = "<unk>";
pub const SEP: &str = "<sep>";

/// Multi-char operators must be matched before their single-char prefixes.
const OPS2: [&str; 3] = ["<=", ">=", "!="];
const PUNCT: [char; 10] = ['=', '<', '>', '"', '\'', ',', '?', '!', '$', '&'];

/// NQL keywords, uppercased on decode when outside a quoted literal.
const NQL_KEYWORDS: [&str; 23] = [
    "from", "as", "of", "where", "and", "search", "order", "by", "asc", "desc",
    "traverse", "trace", "reverse", "limit", "valid", "group", "count", "sum",
    "avg", "min", "max", "true", "false",
];

fn is_keyword(s: &str) -> bool {
    NQL_KEYWORDS.contains(&s)
}

pub struct Tokenizer {
    pub itos: Vec<String>,
    stoi: HashMap<String, u32>,
    pub pad_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
    pub unk_id: u32,
    pub sep_id: u32,
}

impl Tokenizer {
    pub fn new(itos: Vec<String>) -> Self {
        let mut stoi = HashMap::with_capacity(itos.len());
        for (i, t) in itos.iter().enumerate() {
            stoi.insert(t.clone(), i as u32);
        }
        let get = |k: &str| stoi.get(k).copied().unwrap_or(0);
        let pad_id = get(PAD);
        let bos_id = get(BOS);
        let eos_id = get(EOS);
        let unk_id = get(UNK);
        let sep_id = get(SEP);
        Tokenizer { itos, stoi, pad_id, bos_id, eos_id, unk_id, sep_id }
    }

    pub fn len(&self) -> usize {
        self.itos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.itos.is_empty()
    }

    /// Split raw text into atomic pieces. Mirrors Python `pre_tokenize`.
    pub fn pre_tokenize(text: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0usize;
        let mut word = String::new();

        // flush the pending word, splitting digits out of it as the Python does
        fn flush(word: &mut String, out: &mut Vec<String>) {
            if word.is_empty() {
                return;
            }
            if word.chars().any(|c| c.is_ascii_digit()) {
                let mut buf = String::new();
                for ch in word.chars() {
                    if ch.is_ascii_digit() {
                        if !buf.is_empty() {
                            out.push(buf.to_lowercase());
                            buf.clear();
                        }
                        out.push(ch.to_string());
                    } else {
                        buf.push(ch);
                    }
                }
                if !buf.is_empty() {
                    out.push(buf.to_lowercase());
                }
            } else {
                out.push(word.to_lowercase());
            }
            word.clear();
        }

        while i < chars.len() {
            // two-char operators first
            if i + 1 < chars.len() {
                let pair: String = chars[i..i + 2].iter().collect();
                if OPS2.contains(&pair.as_str()) {
                    flush(&mut word, &mut out);
                    out.push(pair);
                    i += 2;
                    continue;
                }
            }
            let c = chars[i];
            if c.is_whitespace() {
                flush(&mut word, &mut out);
                i += 1;
                continue;
            }
            if PUNCT.contains(&c) || c == '.' || c == '-' {
                flush(&mut word, &mut out);
                out.push(c.to_string());
                i += 1;
                continue;
            }
            word.push(c);
            i += 1;
        }
        flush(&mut word, &mut out);
        out
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        Self::pre_tokenize(text)
            .iter()
            .map(|t| self.stoi.get(t).copied().unwrap_or(self.unk_id))
            .collect()
    }

    /// `<s> prompt <sep>` — the inference prefix the model continues from.
    pub fn encode_prompt(&self, prompt: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(prompt.len() / 3 + 4);
        ids.push(self.bos_id);
        ids.extend(self.encode(prompt));
        ids.push(self.sep_id);
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let toks: Vec<&str> = ids
            .iter()
            .filter_map(|&i| self.itos.get(i as usize).map(|s| s.as_str()))
            .filter(|t| ![PAD, BOS, EOS, UNK, SEP].contains(t))
            .collect();
        Self::detok(&toks)
    }

    /// Mirror of Python `CastTokenizer._detok`.
    pub fn detok(toks: &[&str]) -> String {
        let mut out: Vec<String> = Vec::new();
        let mut i = 0usize;
        let mut in_quote = false;

        let is_digit_tok = |s: &str| s.len() == 1 && s.chars().next().unwrap().is_ascii_digit();

        while i < toks.len() {
            let t = toks[i];

            if t == "\"" {
                in_quote = !in_quote;
                out.push(t.to_string());
                i += 1;
                continue;
            }

            // rule 1: merge digit runs, including '-' and '.' before a digit
            let starts_num = is_digit_tok(t)
                || (t == "-" && i + 1 < toks.len() && is_digit_tok(toks[i + 1]));
            if starts_num {
                let mut num = String::from(t);
                i += 1;
                while i < toks.len() {
                    let nx = toks[i];
                    if is_digit_tok(nx) {
                        num.push_str(nx);
                        i += 1;
                        continue;
                    }
                    if (nx == "." || nx == "-")
                        && i + 1 < toks.len()
                        && is_digit_tok(toks[i + 1])
                    {
                        num.push_str(nx);
                        i += 1;
                        continue;
                    }
                    break;
                }
                out.push(num);
                continue;
            }

            // rule 2: keyword casing applies outside quotes only
            if !in_quote && is_keyword(t) {
                out.push(t.to_uppercase());
            } else {
                out.push(t.to_string());
            }
            i += 1;
        }

        let joined = out.join(" ");
        let tightened = tighten_quotes(&joined);
        let tightened = tighten_punct(&tightened);
        tightened.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

/// rule 3: `" paid "` -> `"paid"`. Equivalent to the Python regex
/// `re.sub(r'"\s*([^"]*?)\s*"', ...)`.
fn tighten_quotes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == '"' {
            // find the closing quote
            if let Some(rel) = bytes[i + 1..].iter().position(|&c| c == '"') {
                let close = i + 1 + rel;
                let inner: String = bytes[i + 1..close].iter().collect();
                out.push('"');
                out.push_str(inner.trim());
                out.push('"');
                i = close + 1;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

/// Equivalent to Python `re.sub(r"\s+([,?!])", r"\1", s)`.
fn tighten_punct(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if matches!(ch, ',' | '?' | '!') {
            while out.ends_with(' ') {
                out.pop();
            }
        }
        out.push(ch);
    }
    out
}
