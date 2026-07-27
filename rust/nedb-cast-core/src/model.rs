//! Pure-Rust forward pass, numerically matching `cast/model.py`.
//!
//! Architecture (pre-norm decoder):
//!   x = tok_emb[ids] + pos_emb[0..T]
//!   per block:  x = x + attn(ln1(x));  x = x + mlp(ln2(x))
//!   logits = head(ln_f(x))          with head.weight TIED to tok_emb.weight
//!
//! Conventions that must not drift from PyTorch:
//!   * Linear weights keep (out_features, in_features) layout → `y = x @ Wᵀ + b`
//!   * LayerNorm eps = 1e-5, biased variance (divide by N, not N-1)
//!   * GELU uses the EXACT erf formulation, not the tanh approximation.
//!     `F.gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`. Using tanh here produces
//!     ~1e-3 drift that compounds across 4 layers and breaks greedy decoding on
//!     near-ties.
//!   * Attention softmax is computed in f32 with max-subtraction for stability.

use crate::format::{Config, LoadError, ModelFile};

pub struct Linear {
    pub w: Vec<f32>, // [out, in]
    pub b: Option<Vec<f32>>,
    pub n_in: usize,
    pub n_out: usize,
}

impl Linear {
    /// y[out] = sum_in x[in] * w[out, in] + b[out]
    pub fn forward(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.n_in);
        debug_assert_eq!(y.len(), self.n_out);
        for o in 0..self.n_out {
            let row = &self.w[o * self.n_in..(o + 1) * self.n_in];
            let mut acc = 0f32;
            for i in 0..self.n_in {
                acc += row[i] * x[i];
            }
            y[o] = match &self.b {
                Some(b) => acc + b[o],
                None => acc,
            };
        }
    }
}

pub struct LayerNorm {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn forward(&self, x: &[f32], y: &mut [f32]) {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        // biased variance, matching torch.nn.LayerNorm
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + self.eps).sqrt();
        for i in 0..x.len() {
            y[i] = (x[i] - mean) * inv * self.w[i] + self.b[i];
        }
    }
}

/// Abramowitz & Stegun 7.1.26 is NOT accurate enough here; use the
/// higher-precision rational approximation of erf so GELU tracks PyTorch to
/// ~1e-7. (PyTorch calls into std::erf under the hood.)
fn erf(x: f32) -> f32 {
    // Numerical Recipes erfc-based formulation, double internally for accuracy.
    let z = x as f64;
    let t = 1.0 / (1.0 + 0.5 * z.abs());
    let tau = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    let erfc = if z >= 0.0 { tau } else { 2.0 - tau };
    (1.0 - erfc) as f32
}

#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

pub struct Block {
    pub ln1: LayerNorm,
    pub qkv: Linear,     // [3*C, C]
    pub attn_proj: Linear,
    pub ln2: LayerNorm,
    pub fc: Linear,      // [4C, C]
    pub mlp_proj: Linear, // [C, 4C]
}

pub struct CastModel {
    pub cfg: Config,
    pub tok_emb: Vec<f32>, // [vocab, C]
    pub pos_emb: Vec<f32>, // [block, C]
    pub blocks: Vec<Block>,
    pub ln_f: LayerNorm,
    // head is TIED to tok_emb; no separate storage
}

impl CastModel {
    pub fn from_file(mf: &ModelFile) -> Result<Self, LoadError> {
        let c = mf.config.clone();
        let cn = c.n_embd;

        let lin = |name: &str, n_out: usize, n_in: usize| -> Result<Linear, LoadError> {
            let w = mf.tensor(&format!("{name}.weight"), &[n_out, n_in])?.to_vec();
            let b = if c.bias {
                match mf.tensor(&format!("{name}.bias"), &[n_out]) {
                    Ok(v) => Some(v.to_vec()),
                    Err(LoadError::MissingTensor(_)) => None,
                    Err(e) => return Err(e),
                }
            } else {
                None
            };
            Ok(Linear { w, b, n_in, n_out })
        };
        let ln = |name: &str| -> Result<LayerNorm, LoadError> {
            Ok(LayerNorm {
                w: mf.tensor(&format!("{name}.weight"), &[cn])?.to_vec(),
                b: mf.tensor(&format!("{name}.bias"), &[cn])?.to_vec(),
                eps: 1e-5,
            })
        };

        let tok_emb = mf.tensor("tok_emb.weight", &[c.vocab_size, cn])?.to_vec();
        let pos_emb = mf.tensor("pos_emb.weight", &[c.block_size, cn])?.to_vec();

        let mut blocks = Vec::with_capacity(c.n_layer);
        for i in 0..c.n_layer {
            blocks.push(Block {
                ln1: ln(&format!("blocks.{i}.ln1"))?,
                qkv: lin(&format!("blocks.{i}.attn.attn"), 3 * cn, cn)?,
                attn_proj: lin(&format!("blocks.{i}.attn.proj"), cn, cn)?,
                ln2: ln(&format!("blocks.{i}.ln2"))?,
                fc: lin(&format!("blocks.{i}.mlp.fc"), 4 * cn, cn)?,
                mlp_proj: lin(&format!("blocks.{i}.mlp.proj"), cn, 4 * cn)?,
            });
        }
        let ln_f = ln("ln_f")?;
        Ok(CastModel { cfg: c, tok_emb, pos_emb, blocks, ln_f, })
    }

    /// Full forward pass over a sequence. Returns logits for the LAST position
    /// only (all we need for greedy decoding), and optionally all positions.
    pub fn forward_last(&self, ids: &[u32]) -> Vec<f32> {
        let logits = self.forward(ids, true);
        logits
    }

    /// `only_last = false` returns [T, vocab] flattened; true returns [vocab].
    pub fn forward(&self, ids: &[u32], only_last: bool) -> Vec<f32> {
        let c = &self.cfg;
        let cn = c.n_embd;
        let t = ids.len().min(c.block_size);
        let hs = cn / c.n_head;

        // ---- embeddings
        let mut x = vec![0f32; t * cn];
        for (p, &id) in ids[ids.len() - t..].iter().enumerate() {
            let te = &self.tok_emb[(id as usize) * cn..(id as usize + 1) * cn];
            let pe = &self.pos_emb[p * cn..(p + 1) * cn];
            for i in 0..cn {
                x[p * cn + i] = te[i] + pe[i];
            }
        }

        // scratch
        let mut norm = vec![0f32; cn];
        let mut qkv = vec![0f32; 3 * cn];
        let mut q = vec![0f32; t * cn];
        let mut k = vec![0f32; t * cn];
        let mut v = vec![0f32; t * cn];
        let mut attn_out = vec![0f32; cn];
        let mut proj_out = vec![0f32; cn];
        let mut hidden = vec![0f32; 4 * cn];
        let mut mlp_out = vec![0f32; cn];
        let scale = 1.0 / (hs as f32).sqrt();

        for blk in &self.blocks {
            // ---- attention: project every position first
            for p in 0..t {
                blk.ln1.forward(&x[p * cn..(p + 1) * cn], &mut norm);
                blk.qkv.forward(&norm, &mut qkv);
                q[p * cn..(p + 1) * cn].copy_from_slice(&qkv[0..cn]);
                k[p * cn..(p + 1) * cn].copy_from_slice(&qkv[cn..2 * cn]);
                v[p * cn..(p + 1) * cn].copy_from_slice(&qkv[2 * cn..3 * cn]);
            }
            // ---- causal attention per position
            let mut deltas = vec![0f32; t * cn];
            for p in 0..t {
                for h in 0..c.n_head {
                    let qo = p * cn + h * hs;
                    // scores over 0..=p (causal)
                    let mut scores = vec![0f32; p + 1];
                    let mut maxs = f32::NEG_INFINITY;
                    for s in 0..=p {
                        let ko = s * cn + h * hs;
                        let mut dot = 0f32;
                        for i in 0..hs {
                            dot += q[qo + i] * k[ko + i];
                        }
                        let sc = dot * scale;
                        scores[s] = sc;
                        if sc > maxs {
                            maxs = sc;
                        }
                    }
                    let mut denom = 0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - maxs).exp();
                        denom += *s;
                    }
                    let inv = 1.0 / denom;
                    for i in 0..hs {
                        let mut acc = 0f32;
                        for s in 0..=p {
                            acc += scores[s] * inv * v[s * cn + h * hs + i];
                        }
                        attn_out[h * hs + i] = acc;
                    }
                }
                blk.attn_proj.forward(&attn_out, &mut proj_out);
                deltas[p * cn..(p + 1) * cn].copy_from_slice(&proj_out);
            }
            for i in 0..t * cn {
                x[i] += deltas[i];
            }

            // ---- MLP (position-wise)
            for p in 0..t {
                blk.ln2.forward(&x[p * cn..(p + 1) * cn], &mut norm);
                blk.fc.forward(&norm, &mut hidden);
                for hv in hidden.iter_mut() {
                    *hv = gelu(*hv);
                }
                blk.mlp_proj.forward(&hidden, &mut mlp_out);
                for i in 0..cn {
                    x[p * cn + i] += mlp_out[i];
                }
            }
        }

        // ---- final norm + tied head
        let vocab = c.vocab_size;
        let positions: Vec<usize> = if only_last { vec![t - 1] } else { (0..t).collect() };
        let mut out = vec![0f32; positions.len() * vocab];
        for (oi, &p) in positions.iter().enumerate() {
            self.ln_f.forward(&x[p * cn..(p + 1) * cn], &mut norm);
            for vi in 0..vocab {
                let row = &self.tok_emb[vi * cn..(vi + 1) * cn];
                let mut acc = 0f32;
                for i in 0..cn {
                    acc += row[i] * norm[i];
                }
                out[oi * vocab + vi] = acc;
            }
        }
        out
    }

    /// Greedy decode. Returns the generated ids, excluding EOS.
    pub fn generate(&self, prompt_ids: &[u32], max_new: usize, eos_id: u32) -> Vec<u32> {
        let mut ids = prompt_ids.to_vec();
        let mut gen = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let logits = self.forward(&ids, true);
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for (i, &lv) in logits.iter().enumerate() {
                if lv > bestv {
                    bestv = lv;
                    best = i;
                }
            }
            let nxt = best as u32;
            if nxt == eos_id {
                break;
            }
            gen.push(nxt);
            ids.push(nxt);
            if ids.len() > self.cfg.block_size {
                // slide the window; positions are re-derived in forward()
                ids.remove(0);
            }
        }
        gen
    }
}
