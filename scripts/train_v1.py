#!/usr/bin/env python3
"""Reference training run for v1. Equivalent to:
   python -m cast.cli train --steps 14000
Kept as a script because long runs are usually launched with nohup."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cast.train import train

if __name__ == "__main__":
    r = train(data_dir="data", out_dir="runs/v1", steps=14000, batch_size=32,
              lr=3e-3, warmup=400, eval_every=500, ckpt_every=500,
              n_layer=4, n_embd=256, n_head=4, block_size=128, dropout=0.05,
              threads=2, seed=1337, resume=True)
    os.makedirs("runs/v1", exist_ok=True)
    with open("runs/v1/result.json", "w") as fh:
        json.dump({k: v for k, v in r.items() if k != "history"}, fh, indent=2)
    print(json.dumps({k: v for k, v in r.items() if k != "history"}, indent=2))
