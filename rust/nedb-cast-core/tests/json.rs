//! A minimal JSON reader for the parity fixtures.
//!
//! Exists so `cargo test` needs no dev-dependencies. The fixture shape is fixed
//! and machine-generated, so a straightforward recursive-descent parser is
//! sufficient — this is not meant to be a general-purpose JSON library.

#![allow(dead_code)]

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<JsonValue>),
    Obj(HashMap<String, JsonValue>),
}

#[derive(Debug)]
pub struct ParseError(pub String);

impl JsonValue {
    pub fn parse(s: &str) -> Result<JsonValue, ParseError> {
        let b: Vec<char> = s.chars().collect();
        let mut i = 0usize;
        let v = parse_value(&b, &mut i)?;
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Obj(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            JsonValue::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_f64().map(|f| f as usize)
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        Some(self.as_array()?.iter().filter_map(|v| v.as_f64()).map(|f| f as f32).collect())
    }

    pub fn as_u32_array(&self) -> Option<Vec<u32>> {
        Some(self.as_array()?.iter().filter_map(|v| v.as_f64()).map(|f| f as u32).collect())
    }
}

fn skip_ws(b: &[char], i: &mut usize) {
    while *i < b.len() && b[*i].is_whitespace() {
        *i += 1;
    }
}

fn parse_value(b: &[char], i: &mut usize) -> Result<JsonValue, ParseError> {
    skip_ws(b, i);
    if *i >= b.len() {
        return Err(ParseError("unexpected end".into()));
    }
    match b[*i] {
        '{' => parse_obj(b, i),
        '[' => parse_arr(b, i),
        '"' => Ok(JsonValue::Str(parse_str(b, i)?)),
        't' => {
            expect_lit(b, i, "true")?;
            Ok(JsonValue::Bool(true))
        }
        'f' => {
            expect_lit(b, i, "false")?;
            Ok(JsonValue::Bool(false))
        }
        'n' => {
            expect_lit(b, i, "null")?;
            Ok(JsonValue::Null)
        }
        _ => parse_num(b, i),
    }
}

fn expect_lit(b: &[char], i: &mut usize, lit: &str) -> Result<(), ParseError> {
    for c in lit.chars() {
        if *i >= b.len() || b[*i] != c {
            return Err(ParseError(format!("expected literal {lit}")));
        }
        *i += 1;
    }
    Ok(())
}

fn parse_obj(b: &[char], i: &mut usize) -> Result<JsonValue, ParseError> {
    *i += 1; // {
    let mut m = HashMap::new();
    loop {
        skip_ws(b, i);
        if *i >= b.len() {
            return Err(ParseError("unterminated object".into()));
        }
        if b[*i] == '}' {
            *i += 1;
            break;
        }
        if b[*i] == ',' {
            *i += 1;
            continue;
        }
        let k = parse_str(b, i)?;
        skip_ws(b, i);
        if *i >= b.len() || b[*i] != ':' {
            return Err(ParseError(format!("expected ':' after key {k}")));
        }
        *i += 1;
        let v = parse_value(b, i)?;
        m.insert(k, v);
    }
    Ok(JsonValue::Obj(m))
}

fn parse_arr(b: &[char], i: &mut usize) -> Result<JsonValue, ParseError> {
    *i += 1; // [
    let mut a = Vec::new();
    loop {
        skip_ws(b, i);
        if *i >= b.len() {
            return Err(ParseError("unterminated array".into()));
        }
        if b[*i] == ']' {
            *i += 1;
            break;
        }
        if b[*i] == ',' {
            *i += 1;
            continue;
        }
        a.push(parse_value(b, i)?);
    }
    Ok(JsonValue::Arr(a))
}

fn parse_str(b: &[char], i: &mut usize) -> Result<String, ParseError> {
    skip_ws(b, i);
    if *i >= b.len() || b[*i] != '"' {
        return Err(ParseError("expected string".into()));
    }
    *i += 1;
    let mut out = String::new();
    while *i < b.len() {
        match b[*i] {
            '"' => {
                *i += 1;
                return Ok(out);
            }
            '\\' => {
                *i += 1;
                if *i >= b.len() {
                    break;
                }
                match b[*i] {
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'u' => {
                        let hex: String = b[*i + 1..(*i + 5).min(b.len())].iter().collect();
                        if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                            if let Some(c) = char::from_u32(cp) {
                                out.push(c);
                            }
                        }
                        *i += 4;
                    }
                    other => out.push(other),
                }
                *i += 1;
            }
            c => {
                out.push(c);
                *i += 1;
            }
        }
    }
    Err(ParseError("unterminated string".into()))
}

fn parse_num(b: &[char], i: &mut usize) -> Result<JsonValue, ParseError> {
    let start = *i;
    if *i < b.len() && (b[*i] == '-' || b[*i] == '+') {
        *i += 1;
    }
    while *i < b.len()
        && (b[*i].is_ascii_digit() || b[*i] == '.' || b[*i] == 'e' || b[*i] == 'E'
            || b[*i] == '-' || b[*i] == '+')
    {
        *i += 1;
    }
    let s: String = b[start..*i].iter().collect();
    // Tolerate the non-standard tokens Python's json can emit for specials.
    if s.is_empty() {
        // maybe NaN / Infinity
        for (lit, val) in [
            ("NaN", f64::NAN),
            ("Infinity", f64::INFINITY),
            ("-Infinity", f64::NEG_INFINITY),
        ] {
            let cand: String = b[*i..(*i + lit.len()).min(b.len())].iter().collect();
            if cand == lit {
                *i += lit.len();
                return Ok(JsonValue::Num(val));
            }
        }
        return Err(ParseError("invalid number".into()));
    }
    s.parse::<f64>()
        .map(JsonValue::Num)
        .map_err(|_| ParseError(format!("bad number {s:?}")))
}
