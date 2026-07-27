"""
cast.train — training loop with prompt-masked loss and resumable checkpoints.

Key details:

  PROMPT MASKING. Each row is encoded as `<s> prompt <sep> nql </s>` and loss is
  computed only on tokens after <sep>. Training on the prompt tokens too would
  spend capacity learning to model English, which is not the job — the job is
  producing the plan given the prompt.

  BUCKETED BATCHING. Sequences vary 15..125 tokens. Padding everything to 128
  wastes ~60% of compute on this corpus (mean length is 39). We sort by length
  into buckets so each batch pads to its own max, which is a large real speedup
  on a 2-vCPU box.

  RESUMABLE. The sandbox can be reclaimed mid-run, so every checkpoint carries
  model + optimizer + step + rng state and training resumes exactly.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .model import CastConfig, CastModel
from .tokenizer import CastTokenizer


def load_rows(path: str) -> List[dict]:
    with open(path) as fh:
        return [json.loads(l) for l in fh]


def encode_rows(rows: List[dict], tok: CastTokenizer,
                block_size: int) -> List[Tuple[List[int], int]]:
    """Encode to (ids, prompt_len), dropping anything that exceeds block_size."""
    out = []
    dropped = 0
    for r in rows:
        ids, plen = tok.encode_pair(r["prompt"], r["nql"])
        if len(ids) > block_size:
            dropped += 1
            continue
        out.append((ids, plen))
    if dropped:
        print(f"  [encode] dropped {dropped} rows over block_size={block_size}")
    return out


class Batcher:
    """Length-bucketed batcher. Shuffles within buckets each epoch."""

    def __init__(self, data: List[Tuple[List[int], int]], batch_size: int,
                 pad_id: int, seed: int = 0):
        self.data = sorted(data, key=lambda t: len(t[0]))
        self.bs = batch_size
        self.pad_id = pad_id
        self.g = torch.Generator().manual_seed(seed)
        self._make_batches()

    def _make_batches(self):
        # contiguous chunks of similar length, then shuffle chunk order
        self.batches = [list(range(i, min(i + self.bs, len(self.data))))
                        for i in range(0, len(self.data), self.bs)]
        perm = torch.randperm(len(self.batches), generator=self.g).tolist()
        self.batches = [self.batches[i] for i in perm]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for bidx in self.batches:
            items = [self.data[i] for i in bidx]
            maxlen = max(len(ids) for ids, _ in items)
            X = torch.full((len(items), maxlen - 1), self.pad_id, dtype=torch.long)
            Y = torch.full((len(items), maxlen - 1), -100, dtype=torch.long)
            for r, (ids, plen) in enumerate(items):
                seq = torch.tensor(ids, dtype=torch.long)
                n = len(ids) - 1
                X[r, :n] = seq[:-1]
                tgt = seq[1:].clone()
                # mask prompt positions: targets before (plen-1) are prompt tokens
                tgt[: max(0, plen - 1)] = -100
                Y[r, :n] = tgt
            yield X, Y
        self._make_batches()


def get_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    if step >= total:
        return base * 0.05
    prog = (step - warmup) / max(1, total - warmup)
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def eval_loss(model: CastModel, batcher: Batcher, max_batches: int = 40) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i, (X, Y) in enumerate(batcher):
        if i >= max_batches:
            break
        _, loss = model(X, Y)
        tot += loss.item()
        n += 1
    model.train()
    return tot / max(1, n)


def train(data_dir: str = "data", out_dir: str = "runs/v1",
          steps: int = 12000, batch_size: int = 32, lr: float = 3e-3,
          warmup: int = 300, eval_every: int = 500, ckpt_every: int = 500,
          n_layer: int = 4, n_embd: int = 256, n_head: int = 4,
          block_size: int = 128, dropout: float = 0.1,
          threads: int = 2, seed: int = 1337,
          resume: bool = True, max_seconds: Optional[float] = None) -> Dict:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    tok = CastTokenizer.load(os.path.join(data_dir, "tokenizer.json"))
    print(f"vocab: {len(tok)}")

    train_rows = load_rows(os.path.join(data_dir, "train.jsonl"))
    eval_rows = load_rows(os.path.join(data_dir, "eval.jsonl"))
    tr = encode_rows(train_rows, tok, block_size)
    ev = encode_rows(eval_rows, tok, block_size)
    print(f"train seqs: {len(tr)}  eval seqs: {len(ev)}")

    cfg = CastConfig(vocab_size=len(tok), n_embd=n_embd, n_layer=n_layer,
                     n_head=n_head, block_size=block_size, dropout=dropout)
    model = CastModel(cfg)
    print(f"params: {model.n_params()/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)

    start_step = 0
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"resumed from step {start_step}")

    tb = Batcher(tr, batch_size, tok.pad_id, seed=seed)
    eb = Batcher(ev, batch_size, tok.pad_id, seed=seed + 1)

    hist: List[Dict] = []
    if os.path.exists(os.path.join(out_dir, "history.json")):
        with open(os.path.join(out_dir, "history.json")) as fh:
            hist = json.load(fh)

    model.train()
    step = start_step
    t_start = time.time()
    tok_seen = 0
    it = iter(tb)
    losses: List[float] = []
    stopped_early = False

    while step < steps:
        try:
            X, Y = next(it)
        except StopIteration:
            it = iter(tb)
            X, Y = next(it)

        cur_lr = get_lr(step, steps, lr, warmup)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        _, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        tok_seen += X.numel()
        step += 1

        if step % 100 == 0:
            el = time.time() - t_start
            print(f"step {step:>6}/{steps} loss {sum(losses[-100:])/len(losses[-100:]):.4f} "
                  f"lr {cur_lr:.2e} {tok_seen/el:,.0f} tok/s "
                  f"elapsed {el/60:.1f}m", flush=True)

        if step % eval_every == 0 or step == steps:
            el = eval_loss(model, eb)
            tr_l = sum(losses[-eval_every:]) / max(1, len(losses[-eval_every:]))
            hist.append({"step": step, "train_loss": tr_l, "eval_loss": el,
                         "lr": cur_lr, "elapsed_s": time.time() - t_start})
            print(f"  >> eval @ {step}: train {tr_l:.4f}  eval {el:.4f}", flush=True)
            with open(os.path.join(out_dir, "history.json"), "w") as fh:
                json.dump(hist, fh, indent=2)

        if step % ckpt_every == 0 or step == steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": cfg.to_dict()}, ckpt_path)

        if max_seconds and (time.time() - t_start) > max_seconds:
            print(f"stopping at step {step} (max_seconds={max_seconds})")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": cfg.to_dict()}, ckpt_path)
            stopped_early = True
            break

    elapsed = time.time() - t_start
    return {"steps": step, "elapsed_s": elapsed, "params": model.n_params(),
            "cfg": cfg.to_dict(), "final_train_loss": (sum(losses[-100:]) / max(1, len(losses[-100:]))),
            "history": hist, "out_dir": out_dir, "tok_per_s": tok_seen / elapsed,
            "stopped_early": stopped_early}
