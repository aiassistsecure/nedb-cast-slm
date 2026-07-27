"""
cast.model — the nedb-cast-slm decoder.

A small causal transformer, sized from throughput measured on this box (2 vCPU,
no GPU): d=256, L=4, heads=4, T=128 → ~3.7M params at ~2800 tok/s.

Choices that matter at this scale:
  - WEIGHT TYING (head.weight = token_embedding.weight). With a 516-token vocab
    this saves little memory, but it consistently helps small models generalise
    and removes a whole output matrix from the gradient path.
  - PRE-NORM blocks (LayerNorm before attention/MLP). More stable than post-norm
    when training from scratch without a warmup budget to spare.
  - LEARNED positional embeddings. Sequences are short and fixed-max (128); rope
    or ALiBi buys nothing here and costs clarity.
  - GELU MLP at 4x. Standard, and the MLP is where most of the capacity lives.

Everything is plain torch — no transformers dependency — so the published package
stays tiny and has no version-drift surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CastConfig:
    vocab_size: int = 516
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 4
    block_size: int = 128
    dropout: float = 0.1
    bias: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: CastConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop_p = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.attn(x).split(self.n_embd, dim=2)
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # scaled_dot_product_attention handles the causal mask efficiently and
        # is meaningfully faster than a hand-rolled masked matmul on CPU.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_drop_p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: CastConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: CastConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class CastModel(nn.Module):
    def __init__(self, cfg: CastConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init)
        # scaled init on residual projections (GPT-2 style) keeps deep-ish
        # residual stacks from blowing up early
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"seq {T} > block {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            # ignore_index=-100 is how prompt tokens and padding are masked out,
            # so loss is computed ONLY on the NQL continuation
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 eos_id: Optional[int] = None,
                 temperature: float = 0.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """Autoregressive decode. temperature=0 → greedy (what we want for a DSL)."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
